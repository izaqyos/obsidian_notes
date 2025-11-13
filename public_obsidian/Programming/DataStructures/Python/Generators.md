Here’s a fast, practical 101 on Python generators.

# What they are

- **Generators** are functions that use `yield` to produce a sequence of values **lazily** (on demand).
    
- Each call to `next()` runs the function until the next `yield`, **pausing** and preserving local state between yields.
    
- They’re **iterators**, so you can loop over them with `for`.
    

```python
def countdown(n):
    while n > 0:
        yield n
        n -= 1

for x in countdown(3):
    print(x)  # 3, 2, 1
```

# Why use them

- **Memory-efficient:** don’t build big lists in RAM.
    
- **Composable pipelines:** chain filters/transforms without intermediates.
    
- **Infinite / streaming data:** natural fit for unbounded sequences.
    

# Core patterns

### Basic generator function

```python
def squares(n):
    for i in range(n):
        yield i * i

list(squares(5))  # [0, 1, 4, 9, 16]
```

### Generator expressions (like list comps, but lazy)

```python
squares = (x*x for x in range(10))
sum_even_squares = sum(x for x in squares if x % 2 == 0)
```

### Pipelining

```python
lines = (line.strip() for line in open("data.txt"))
nums = (int(s) for s in lines if s)
total = sum(n for n in nums if n > 0)
```

### `yield from` (delegate to a sub-iterator)

```python
def flatten(nested):
    for sub in nested:
        yield from sub

list(flatten([[1,2],[3],[4,5]]))  # [1,2,3,4,5]
```

# A bit more advanced

### Sending values in (`send`) and closing

```python
def accumulator():
    total = 0
    while True:
        val = yield total  # receives value from .send()
        if val is None:
            return total    # StopIteration.value holds this
        total += val

gen = accumulator()
next(gen)         # prime -> yields 0
gen.send(10)      # -> 10
gen.send(5)       # -> 15
try:
    gen.send(None)
except StopIteration as e:
    print(e.value)  # 15
```

# Common gotchas

- **One-shot:** once a generator is exhausted, it’s done. Recreate it if you need to iterate again.
    
- **No random access:** no `len()`, indexing, or slicing (without `itertools.islice`).
    
- **Side effects happen on iteration:** if iteration stops early, cleanup code should be in `finally` blocks or use context managers.
    

# Handy tools

- `itertools` module: `islice`, `chain`, `takewhile`, `cycle`, `accumulate`, etc.—all play nicely with generators.
    

awesome—let’s level up. here are advanced generator techniques and patterns with compact, runnable examples.

# mechanics beyond `yield`

### `send`, `throw`, `close` (the full protocol)

```python
def logger():
    try:
        msg = "ready"
        while True:
            cmd = yield msg          # receives via .send()
            if cmd == "ping":
                msg = "pong"
            else:
                msg = f"echo:{cmd}"
    except GeneratorExit:
        print("logger closed")
    except Exception as e:           # .throw() lands here
        print("logger error:", e)
        yield "error-handled"

g = logger()
next(g)                 # 'ready'
g.send("ping")          # 'pong'
g.throw(ValueError("boom"))  # prints, yields 'error-handled'
g.close()               # prints 'logger closed'
```

### `yield from`: delegation + return values (PEP 380)

- Delegates all `next/send/throw/close` to a subgenerator.
    
- Captures the subgenerator’s **return value** (via `return x` inside the subgenerator).
    

```python
def sum_until_none():
    total = 0
    while True:
        v = yield
        if v is None:
            return total    # becomes StopIteration.value

def grouped_sum(groups):
    for group in groups:
        acc = sum_until_none()
        next(acc)                  # prime
        for x in group:
            acc.send(x)
        total = yield from acc     # collect return
        yield total

list(grouped_sum([[1,2,3], [10]]))   # [6, 10]
```

# patterns you’ll actually use

### 1) Generator-based coroutine with auto-priming

```python
def coroutine(fn):
    def wrapped(*a, **k):
        g = fn(*a, **k)
        next(g)
        return g
    return wrapped

@coroutine
def moving_avg(window=3):
    from collections import deque
    q, s = deque(maxlen=window), 0
    while True:
        x = yield (s/len(q) if q else None)
        if isinstance(x, int):      # data point
            if len(q) == q.maxlen: s -= q[0]
            q.append(x); s += x
        elif isinstance(x, tuple) and x[0] == "set_window":
            q = type(q)(maxlen=x[1]); s = sum(q)

avg = moving_avg(3)
avg.send(5)    # -> None
avg.send(7)    # -> 6.0
avg.send(4)    # -> 5.333...
avg.send(("set_window", 2))  # -> 5.5 (recomputed)
```

### 2) Streaming I/O with backpressure (sentinel form)

```python
from functools import partial

def read_chunks(path, size=8192):
    with open(path, "rb") as f:
        for chunk in iter(partial(f.read, size), b""):
            yield chunk

def sha256_of_file(path):
    import hashlib
    h = hashlib.sha256()
    for chunk in read_chunks(path):
        h.update(chunk)
    return h.hexdigest()
```

- `iter(callable, sentinel)` repeatedly calls until the sentinel (`b""`) appears—no extra buffers.
    

### 3) Sliding windows lazily

```python
from collections import deque

def windowed(iterable, size):
    it = iter(iterable)
    d = deque([], maxlen=size)
    for _ in range(size):
        d.append(next(it))
    yield tuple(d)
    for x in it:
        d.append(x)
        yield tuple(d)

list(windowed(range(6), 3))  # [(0,1,2),(1,2,3),(2,3,4),(3,4,5)]
```

### 4) Recursive tree walk with `yield from`

```python
def walk(node):
    # node: {"value": x, "children": [...]}
    yield node["value"]
    for c in node.get("children", []):
        yield from walk(c)
```

### 5) On-the-fly merge (round-robin) of multiple iterables

```python
from collections import deque

def round_robin(*iterables):
    iters = deque(map(iter, iterables))
    while iters:
        it = iters.popleft()
        try:
            yield next(it)
            iters.append(it)
        except StopIteration:
            pass

list(round_robin("ABC", [1,2], ("x","y","z")))  # ['A',1,'x','B',2,'y','C','z']
```

### 6) Generator-based context managers (`@contextmanager`)

```python
import time
from contextlib import contextmanager

@contextmanager
def timed(label="block"):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        print(label, "took", time.perf_counter() - t0, "s")

with timed("heavy"):
    sum(range(5_000_000))
```

### 7) Incremental parsing via coroutines (state machine-ish)

```python
def numbers_only():
    total = 0
    try:
        while True:
            chunk = yield total
            for tok in chunk.split():
                if tok.isdigit():
                    total += int(tok)
    except GeneratorExit:
        pass

g = numbers_only(); next(g)
g.send("10 spam 20")   # 30
g.send("5 eggs")       # 35
g.close()
```

### 8) Robust cleanup with `try/finally`

```python
def resourceful():
    print("open")
    try:
        while True:
            yield "tick"
    finally:
        print("close")

g = resourceful()
next(g)   # prints 'open'
g.close() # prints 'close' even if you didn’t exhaust it
```

### 9) BFS over a graph (lazy, memory-friendly frontier)

```python
from collections import deque

def bfs(start, neighbors):
    seen = {start}
    q = deque([start])
    while q:
        v = q.popleft()
        yield v
        for w in neighbors(v):
            if w not in seen:
                seen.add(w); q.append(w)
```

### 10) Composable “pull” pipelines

```python
def grep(pattern, lines):
    for line in lines:
        if pattern in line:
            yield line

def lower(lines):
    for line in lines:
        yield line.lower()

def uniq(lines):
    prev = object()
    for line in lines:
        if line != prev:
            yield line
            prev = line

pipeline = uniq(lower(grep("error", open("app.log"))))
for line in pipeline:
    ...
```

# async generators (modern coroutines)

Use when you need `await` inside the producer.

```python
import asyncio

async def ticker(n, dt=0.1):
    for i in range(n):
        await asyncio.sleep(dt)
        yield i

async def main():
    async for t in ticker(3):
        print(t)

# asyncio.run(main())
```

- `async def ...: yield` → **async generator**.
    
- Consumed with `async for`.
    
- Great for streaming sockets, websockets, or event feeds.
    

# gotchas & pro tips

- **Priming**: a generator can’t receive with `.send(x)` until after the first `yield`. Use a small decorator (shown above) or call `next(gen)` once.
    
- **One pass only**: generators are exhausted after iteration. If you need multiple passes, regenerate or use `itertools.tee` (careful: `tee` buffers).
    
- **Exceptions**: prefer `.close()`/`with` patterns so `finally` blocks run even if a consumer bails early.
    
- **`yield from` > manual loops**: it forwards `send/throw/close` automatically and makes code clearer.
    
- **Testing**: bake tiny prints/asserts to verify protocol flows (`next`, `send`, `throw`, `close`).
    
