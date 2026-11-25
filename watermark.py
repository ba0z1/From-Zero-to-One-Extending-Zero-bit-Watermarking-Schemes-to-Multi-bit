"""Six-layer token watermark with indexed blocks and hash-chain verification."""

import hashlib
import hmac
import struct
from dataclasses import dataclass, field

import numpy as np
import torch


DEPTH = 6
PARTITION_SEED = b"tdsc-single-text-teaching-8640"
PRF_DOMAIN = b"TDSC-INDEXED-V1\x00PAYLOAD\x00"
PAD_TOKEN = 2**32 - 1


@dataclass(frozen=True)
class Config:
    vocab_size: int
    hash_bits: int = 16
    threshold_bits: int = 32

    def __post_init__(self):
        if type(self.vocab_size) is not int or not 33 <= self.vocab_size <= 2**20:
            raise ValueError("vocab_size must be an integer between 33 and 1048576")
        if type(self.hash_bits) is not int or not 1 <= self.hash_bits <= 256:
            raise ValueError("hash_bits must be an integer between 1 and 256")
        if type(self.threshold_bits) is not int or not 1 <= self.threshold_bits <= 256:
            raise ValueError("threshold_bits must be an integer between 1 and 256")


def derive_response_key(master_key, nonce):
    if len(master_key) != 32 or len(nonce) != 16:
        raise ValueError("Expected a 32-byte master key and a 16-byte response nonce")
    return hmac.digest(master_key, b'SK\x00' + nonce, 'sha256')


def derive_payload_key(master_key, nonce):
    return hmac.digest(derive_response_key(master_key, nonce), b"LWS-CUSTOM-PAYLOAD-v1", "sha256")


def digest_ids(token_ids, domain, hash_bits):
    data = domain + struct.pack(">Q", len(token_ids))
    data += b"".join(struct.pack(">I", token) for token in token_ids)
    digest = hashlib.sha256(data).digest()
    return "".join(f"{byte:08b}" for byte in digest)[:hash_bits]


def validate_ids(ids, vocab_size):
    if not isinstance(ids, list) or any(type(i) is not int or not 0 <= i < vocab_size for i in ids):
        raise ValueError("Expected a list of integer token IDs within the vocabulary")


def partition(index, vocab_size):
    modulus = 2**63 - 1
    seed = 1469598103934665603
    for byte in PARTITION_SEED:
        seed = ((seed ^ byte) * 1099511628211) % modulus
    seed = (seed + (index + 1) * 6364136223846793005) % modulus
    generator = torch.Generator(device="cpu").manual_seed(seed)
    width = (vocab_size - 1).bit_length()
    padded_size = 1 << width
    valid = torch.randperm(vocab_size, generator=generator, dtype=torch.long)
    return torch.cat((valid, torch.arange(vocab_size, padded_size, dtype=torch.long)))


def reserve_contexts(history, seen):
    previous = [PAD_TOKEN] * max(0, 4 - len(history)) + history[-4:]
    contexts = [b'CTX4\x00' + struct.pack(">4IH", *previous, layer) for layer in range(DEPTH)]
    eligible = [ctx not in seen for ctx in contexts]
    seen.update(contexts)
    return contexts, eligible


def prf_integer(key, q, context):
    message = PRF_DOMAIN + struct.pack(">Q", q) + context
    return int.from_bytes(hmac.digest(key, message, 'sha256')[:16], "big")


def wat_sampler(p0, wanted, value):
    if p0 == 0 or p0 == 1:
        return int(p0 == 0)
    probability = p0 if wanted == 0 else 1 - p0
    numerator, denominator = float(probability).as_integer_ratio()
    below = (2 * value + 1) * denominator <= numerator * (1 << 129)
    return wanted if below else 1 - wanted


def wat_detector(branch, value):
    return int(value < (1 << 127)) ^ (1 - branch)


def criterion(n, ones):
    return np.exp(-2 * ((n - 2 * ones) / n) ** 2 * n)


@dataclass
class Sender:
    prompt_ids: list
    response_key: bytes = field(repr=False)
    config: Config
    payload: str = field(default=None, repr=False)
    token_ids: list = field(default_factory=list, init=False)
    blocks: list = field(default_factory=list, init=False)
    seen: set = field(default_factory=set, init=False, repr=False)
    link_start: int = field(default=0, init=False)
    block_start: int = field(default=0, init=False)
    votes: int = field(default=0, init=False)
    ones: int = field(default=0, init=False)

    def __post_init__(self):
        validate_ids(self.prompt_ids, self.config.vocab_size)
        if self.payload is not None:
            if not isinstance(self.payload, str) or not 1 <= len(self.payload) <= 8192 or set(self.payload) - {"0", "1"}:
                raise ValueError("Custom payload must contain 1 to 8192 binary digits")
            self.target = self.payload
        else:
            self.target = digest_ids(self.prompt_ids, b'z\x00', self.config.hash_bits)

    def embed_token(self, probabilities):
        """Draw one token from the model's normalized, 1-D next-token probabilities."""
        vocab = self.config.vocab_size
        if probabilities.shape != (vocab,):
            raise ValueError("Probability vector and vocabulary sizes differ")
        if self.payload is not None and len(self.blocks) == len(self.payload):
            token = int(torch.multinomial(probabilities, 1).item())
            self.token_ids.append(token)
            return token
        index = len(self.token_ids)
        contexts, eligible = reserve_contexts(self.token_ids, self.seen)
        order = partition(index, vocab)
        padded = torch.nn.functional.pad(probabilities, (0, len(order) - vocab))
        ordered = padded[order.to(probabilities.device)]
        low, high = 0, len(order)
        link_complete = False
        for layer in range(DEPTH):
            split = (low + high + 1) // 2
            left_mass = float(ordered[low:split].sum().item())
            p0 = left_mass / float(ordered[low:high].sum().item())
            if p0 > 1:
                left = float(ordered[low:split].sum(dtype=torch.float64).item())
                right = float(ordered[split:high].sum(dtype=torch.float64).item())
                p0 = left / (left + right)
            q = len(self.blocks) + 1
            if eligible[layer]:
                value = prf_integer(self.response_key, q, contexts[layer])
                branch = wat_sampler(p0, int(self.target[(q - 1) % len(self.target)]), value)
                self.votes += 1
                self.ones += wat_detector(branch, value)
            else:
                weights = probabilities.new_tensor([p0, 1 - p0])
                branch = int(torch.multinomial(weights, 1).item())
            if branch == 0:
                high = split
            else:
                low = split
            if eligible[layer] and criterion(self.votes, self.ones) <= 2.0 ** -self.config.threshold_bits:
                end = index * DEPTH + layer + 1
                self.blocks.append(dict(q=q, start=self.block_start, end=end, bit=int(2 * self.ones > self.votes)))
                self.votes = self.ones = 0
                self.block_start = end
                if q % len(self.target) == 0:
                    link_complete = True
                    break
        group = ordered[low:high]
        offset = int(torch.multinomial(group / group.sum(), 1).item())
        token = int(order[low + offset])
        self.token_ids.append(token)
        if link_complete and self.payload is None:
            self.target = digest_ids(self.token_ids[self.link_start:], b'c\x00', self.config.hash_bits)
            self.link_start = len(self.token_ids)
            self.block_start = self.link_start * DEPTH
        return token


class Receiver:
    def __init__(self, received_ids, response_key, config):
        validate_ids(received_ids, config.vocab_size)
        self.config, self.key = config, response_key
        self.size = len(received_ids) * DEPTH
        self.cache = {}
        contexts, bits, positions, seen, history = [], [], [], set(), []
        width = (config.vocab_size - 1).bit_length()
        for index, token in enumerate(received_ids):
            ctx, eligible = reserve_contexts(history, seen)
            order = partition(index, config.vocab_size)
            rank = int(torch.nonzero(order == token).item())
            for layer, use in enumerate(eligible):
                if use:
                    positions.append(index * DEPTH + layer)
                    contexts.append(ctx[layer])
                    bits.append((rank >> (width - layer - 1)) & 1)
            history.append(token)
        self.positions = np.array(positions, dtype=np.int64)
        self.contexts, self.bits = contexts, bits

    def detect_block(self, q, start, cap=None):
        stop = self.size if cap is None else min(self.size, start + cap)
        first, last = np.searchsorted(self.positions, [start, stop])
        if q not in self.cache:
            self.cache[q] = np.full(len(self.positions), -1, dtype=np.int8)
        votes = self.cache[q]
        n, ones = 0, 0
        for i in range(int(first), int(last), 256):
            batch = votes[i:min(i + 256, int(last))]
            for offset in np.flatnonzero(batch < 0):
                j = i + offset
                batch[offset] = wat_detector(self.bits[j], prf_integer(self.key, q, self.contexts[j]))
            counts = n + np.arange(1, len(batch) + 1, dtype=np.int64)
            sums = ones + np.cumsum(batch, dtype=np.int64)
            hits = np.flatnonzero(criterion(counts, sums) <= 2.0 ** -self.config.threshold_bits)
            if len(hits):
                k = hits[0]
                return dict(q=q, start=start, end=int(self.positions[i + k]) + 1,
                            bit=int(2 * sums[k] > counts[k]))
            n, ones = int(counts[-1]), int(sums[-1])
        return None

    def decode(self, window=8, span=6000, budget=10000, payload_length=None):
        if window < 0 or span < 1 or budget < 0:
            raise ValueError("Invalid search limits")
        blocks, start, q, used = [], 0, 1, 0
        while start < self.size:
            block = self.detect_block(q, start)
            mode = "seq"
            if block is None and window:
                mode = "resync"
                for candidate_start in range(start, self.size):
                    last_q = q + window if payload_length is None else min(q + window, payload_length)
                    for candidate_q in range(q, last_q + 1):
                        if (candidate_start, candidate_q) == (start, q):
                            continue
                        if used == budget:
                            return blocks, True
                        used += 1
                        block = self.detect_block(candidate_q, candidate_start, span)
                        if block is not None:
                            break
                    if block is not None:
                        break
            if block is None:
                break
            block["mode"] = mode
            blocks.append(block)
            q = block["q"] + 1
            if payload_length is None:
                start = next_start(block, self.config.hash_bits)
            elif q > payload_length:
                break
            else:
                start = block["end"]
            self.cache = {k: v for k, v in self.cache.items() if k >= q}
        return blocks, False


def next_start(block, hash_bits):
    end = block["end"]
    return ((end + DEPTH - 1) // DEPTH) * DEPTH if block["q"] % hash_bits == 0 else end


def extract_payload(received_ids, response_key, config, payload_length, *, window=8):
    if type(payload_length) is not int or not 1 <= payload_length <= 8192:
        raise ValueError("Expected a payload length between 1 and 8192 bits")
    receiver = Receiver(received_ids, response_key, config)
    blocks, exhausted = receiver.decode(window=window, payload_length=payload_length)
    bits = ["?"] * payload_length
    for block in blocks:
        bits[block["q"] - 1] = str(block["bit"])
    return dict(status="COMPLETE" if len(blocks) == payload_length else "PARTIAL",
                recovered_bits="".join(bits), recovered_count=len(blocks),
                blocks=blocks, search_budget_exhausted=exhausted)


def verify(prompt_ids, received_ids, response_key, config, *, window=8, span=6000, budget=10000):
    """Recover bits and authenticate the longest continuous prefix of whole links."""
    validate_ids(prompt_ids, config.vocab_size)
    receiver = Receiver(received_ids, response_key, config)
    blocks, exhausted = receiver.decode(window, span, budget)
    h = config.hash_bits
    expected = digest_ids(prompt_ids, b'z\x00', h)
    start, q, carrier_start, endpoint = 0, 1, 0, None
    status = "INSUFFICIENT"
    matched_links = 0
    for offset in range(0, len(blocks) - h + 1, h):
        group = blocks[offset:offset + h]
        for block in group:
            if (block["q"], block["start"], block["mode"]) != (q, start, "seq"):
                break
            q += 1
            start = next_start(block, h)
        else:
            if "".join(str(b["bit"]) for b in group) != expected:
                status = "MISMATCH"
                break
            end = (group[-1]["end"] + DEPTH - 1) // DEPTH
            endpoint = carrier_start
            expected = digest_ids(received_ids[carrier_start:end], b'c\x00', h)
            carrier_start = end
            matched_links += 1
            status = "OK"
            continue
        status = "INSUFFICIENT"
        break
    return dict(status=status, authenticated_prefix_tokens=endpoint, matched_links=matched_links,
                recovered_bits="".join(str(b["bit"]) for b in blocks),
                blocks=blocks, search_budget_exhausted=exhausted)
