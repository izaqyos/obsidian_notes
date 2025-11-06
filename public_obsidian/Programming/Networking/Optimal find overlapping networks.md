Treat IPv4 networks as integer intervals and reuse the **RangeSet** mechanics. We’ll:

1. Convert `CIDR` (e.g., `10.192.0.0/24`) → integer **half-open** range `[start, end)` where  
    `start = network.network_address`, `end = broadcast_address + 1`.
    
2. Keep everything in `0 … 2^32`.
    
3. Provide helpers to add/remove CIDRs, test overlaps, and query with IPs or CIDRs.
    

---

## Code: IPv4-aware RangeSet

```python
from bisect import bisect_left
from ipaddress import IPv4Network, IPv4Address
from typing import List, Tuple, Iterable, Iterator

# ---------- IPv4 <-> int helpers ----------

MAX32 = 1 << 32  # 2^32

def ip_to_int(ip: str | IPv4Address) -> int:
    return int(IPv4Address(ip))

def int_to_ip(x: int) -> str:
    if not (0 <= x < MAX32):
        raise ValueError("IPv4 integer out of range")
    return str(IPv4Address(x))

def cidr_to_range(cidr: str | IPv4Network) -> Tuple[int, int]:
    """
    Return half-open [start, end) for the IPv4 network.
    Example: '10.192.0.0/24' -> [10.192.0.0, 10.192.1.0)
    """
    net = IPv4Network(cidr, strict=False) if not isinstance(cidr, IPv4Network) else cidr
    start = int(net.network_address)
    last  = int(net.broadcast_address)
    end   = min(MAX32, last + 1)  # half-open; clamp at 2^32 to avoid overflow on /0
    return start, end

# ---------- Base RangeSet on integers (half-open [l,r)) ----------

Interval = Tuple[int, int]

class RangeSet:
    """
    Disjoint, sorted half-open intervals [l, r) over integers.
    Amortized O(log n + k) per add/remove (k merged/split segments).
    """
    __slots__ = ("_iv", "merge_adjacent")

    def __init__(self, intervals: Iterable[Interval] = (), *, merge_adjacent: bool = True):
        self._iv: List[Interval] = []
        self.merge_adjacent = merge_adjacent
        for l, r in intervals:
            self.add(l, r)

    def _find_by_end_ge(self, x: int) -> int:
        ends = [e for _, e in self._iv]
        return bisect_left(ends, x)

    def add(self, L: int, R: int) -> None:
        if L >= R:
            return
        i = self._find_by_end_ge(L)
        l, r = L, R
        if i > 0:
            pl, pr = self._iv[i-1]
            if pr > L or (self.merge_adjacent and pr == L):
                i -= 1
                l = min(l, pl); r = max(r, pr)
        j = i
        while j < len(self._iv):
            a, b = self._iv[j]
            if a > r or (a == r and not self.merge_adjacent):
                break
            r = max(r, b); j += 1
        self._iv[i:j] = [(l, r)]

    def remove(self, L: int, R: int) -> None:
        if L >= R or not self._iv:
            return
        i = self._find_by_end_ge(L)
        out: List[Interval] = []
        k = i - 1 if (i > 0 and self._iv[i-1][1] > L) else i
        t = k
        while t < len(self._iv):
            s, e = self._iv[t]
            if s >= R:
                break
            if s < L: out.append((s, min(e, L)))
            if e > R: out.append((max(R, s), e))
            t += 1
        self._iv[k:t] = out

    def overlaps(self, L: int, R: int) -> bool:
        if L >= R or not self._iv:
            return False
        i = self._find_by_end_ge(R)
        if i > 0 and self._iv[i-1][1] > L: return True
        return i < len(self._iv) and self._iv[i][0] < R

    def contains_point(self, x: int) -> bool:
        if not self._iv: return False
        i = self._find_by_end_ge(x)
        if i > 0:
            l, r = self._iv[i-1]
            return l <= x < r
        return False

    def total_covered(self) -> int:
        return sum(r - l for l, r in self._iv)

    def iter_segments(self) -> Iterator[Interval]:
        yield from self._iv

# ---------- IPv4 wrapper ----------

class IPv4RangeSet:
    """
    RangeSet specialized for IPv4 addresses.
    Stores disjoint half-open integer ranges but exposes IP/CIDR methods.
    """
    def __init__(self, merge_adjacent: bool = True):
        self._rs = RangeSet(merge_adjacent=merge_adjacent)

    # --- add/remove ---

    def add_cidr(self, cidr: str | IPv4Network) -> None:
        s, e = cidr_to_range(cidr)
        self._rs.add(s, e)

    def remove_cidr(self, cidr: str | IPv4Network) -> None:
        s, e = cidr_to_range(cidr)
        self._rs.remove(s, e)

    def add_ip_range(self, ip_start: str, ip_end_inclusive: str) -> None:
        s = ip_to_int(ip_start)
        e = ip_to_int(ip_end_inclusive)
        if e < s: return
        self._rs.add(s, min(MAX32, e + 1))

    def remove_ip_range(self, ip_start: str, ip_end_inclusive: str) -> None:
        s = ip_to_int(ip_start)
        e = ip_to_int(ip_end_inclusive)
        if e < s: return
        self._rs.remove(s, min(MAX32, e + 1))

    # --- queries ---

    def overlaps_cidr(self, cidr: str | IPv4Network) -> bool:
        s, e = cidr_to_range(cidr)
        return self._rs.overlaps(s, e)

    def contains_ip(self, ip: str | IPv4Address) -> bool:
        return self._rs.contains_point(ip_to_int(ip))

    def total_covered_ips(self) -> int:
        return self._rs.total_covered()

    def segments(self) -> List[Tuple[str, str]]:
        """
        Return current union as list of [start_ip, end_ip_inclusive] in strings.
        """
        out: List[Tuple[str, str]] = []
        for s, e in self._rs.iter_segments():
            start_ip = int_to_ip(s)
            end_ip_incl = int_to_ip(e - 1)  # convert half-open to inclusive
            out.append((start_ip, end_ip_incl))
        return out

    def overlaps_ip_range(self, ip_start: str, ip_end_inclusive: str) -> bool:
        s = ip_to_int(ip_start)
        e = ip_to_int(ip_end_inclusive)
        if e < s: return False
        return self._rs.overlaps(s, min(MAX32, e + 1))
```

---

## Examples

```python
S = IPv4RangeSet()

# Add CIDRs
S.add_cidr("10.192.0.0/24")    # 10.192.0.0 .. 10.192.0.255
S.add_cidr("10.192.1.0/25")    # 10.192.1.0  .. 10.192.1.127

assert S.contains_ip("10.192.0.42")
assert not S.contains_ip("10.192.2.1")

# Overlap checks
assert S.overlaps_cidr("10.192.0.128/25")
assert not S.overlaps_cidr("10.192.2.0/24")

# Add/remove arbitrary IP ranges
S.add_ip_range("10.192.1.128", "10.192.1.191")  # fills rest of /25+
S.remove_ip_range("10.192.0.240", "10.192.0.255")

# Inspect union
print(S.segments())
# -> e.g. [('10.192.0.0', '10.192.0.239'), ('10.192.1.0', '10.192.1.191')]

# Total covered IPs
print(S.total_covered_ips())   # 240 + 192 = 432
```

---

## Notes / pitfalls (handled above)

- **Half-open `[start, end)`** internally; when converting to strings we output inclusive end for readability.
    
- **/32** becomes a single IP range where `end = start + 1` → works naturally.
    
- **/0** (`0.0.0.0/0`) clamps `end` at `2^32` to avoid overflow.
    
- **Non-CIDR arbitrary IP ranges** supported via `add_ip_range/remove_ip_range`.
    
- **Adjacency policy**: `merge_adjacent=True` will merge `…255` + next `…0` into one segment when they touch (useful for de-fragmentation). Set `False` if you must keep fenceposts.
    
