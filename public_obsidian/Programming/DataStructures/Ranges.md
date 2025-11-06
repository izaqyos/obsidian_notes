
![[Pasted image 20251026175917.png]]


![[Pasted image 20251026180216.png]]

Here’s the no-BS **cheat sheet** for working with a list/array of ranges (intervals). Use **half-open** `[L, R)` semantics unless you have a very good reason not to.

---

# 1) Pick-by-operation matrix

|You need to…|Static set (no updates)|Dynamic inserts/deletes|Returns|Time (typical)|
|---|---|---|---|---|
|**Does anything overlap [L,R)?**|Sort starts/ends + binary search|**RangeSet** (coalesced union)|boolean|`O(log n)`|
|**List all intervals overlapping [L,R)**|Pre-sort + sweep offline|**Interval Tree** (BST with `max_end`)|IDs/intervals|`O(log n + k)`|
|**Point stabbing: how many cover x?**|Starts/ends binary search|**Fenwick (BIT)** or **Segment Tree** (with compression)|count|`O(log n)`|
|**Maintain union length**|One pass merge|**RangeSet** or **Difference Map** (scan to compute)|scalar|`O(log n)` per update; union length recompute `O(m)` or keep running if needed|
|**Range add/assign + range sum/min/max on dense indexes**|Build Segment Tree|**Segment Tree (lazy)**|aggregate|`O(log n)`|
|**Just overlap counts at any x / k-booking**|Sweep line|**Difference Map** (endpoint deltas) or BIT|counts / max k|`O(log n)` per update (ordered map/BIT)|
|**Piecewise labels (interval → value), split/merge**|—|**RangeMap** (ordered map keyed by start)|segments w/ values|ops `O(log n + splits)`|

**Static** = build once, query many; **Dynamic** = frequent add/remove.

---

# 2) What each structure is actually for

- **RangeSet (union of intervals)**: keep disjoint, sorted segments. Fast overlap checks, membership tests, union length, subtract/add ranges. Perfect when you don’t care _which original_ interval overlapped—only the union.
    
    - Ops: `add`, `remove`, `overlaps([L,R))`, `contains(x)`, `total_covered()`.
        
    - Cost: `O(log n + m)` per add/remove (`m` = merged/split neighbors).
        
- **Interval Tree (augmented BST with `max_end`)**: the canonical index for “find all intervals overlapping X”.
    
    - Ops: insert/delete interval; `search_all_overlaps([L,R))`; stabbing queries.
        
    - Cost: `O(log n + k)` query (balanced); `O(log n)` update.
        
- **Fenwick (BIT) / Segment Tree** (after **coordinate compression**):
    
    - Use when the domain is (or can be compressed to) integers and you need **counts/aggregates** over points or ranges.
        
    - BIT: smaller, simpler; great for **point add + prefix sum** or **range add + point query**.
        
    - Segment Tree (lazy): rich **range update + range query** (sum/min/max).
        
    - Cost: `O(log n)` per op.
        
- **Difference Map (endpoint deltas)**: maintain a map of `+1@L`, `-1@R`. A prefix over sorted keys gives active overlap count.
    
    - Use for k-booking, max overlap, total union length (scan where count>0).
        
    - Dead simple; dynamic but you’ll recompute via scan unless you keep an ordered tree to maintain a running prefix.
        
- **RangeMap** (piecewise constant): like RangeSet but each segment carries a value (label). Insertion splits existing segments; adjacent equal values merge.
    
    - Use for “ACLs/feature flags/price bands across a line”.
        
- **Sweep Line (offline/batch)**: sort endpoints; walk once. Best when the whole workload is known up front (no online updates).
    

---

# 3) Decision recipe (fast)

1. **Do you need identities (which intervals overlap)?**
    
    - **Yes** → **Interval Tree**.
        
    - **No** → go to 2.
        
2. **Only need boolean coverage/union length with add/remove?**
    
    - **RangeSet**. (If you also need labels → **RangeMap**.)
        
3. **Need counts/aggregates at points or ranges on an indexable axis?**
    
    - Discrete or compressible → **Fenwick** (simple) or **Segment Tree** (rich ops).
        
    - Not discrete and only k-booking/union metrics → **Difference Map**.
        
4. **One-shot batch analytics?**
    
    - **Sweep line** + optionally DSU for clustering.
        

---

# 4) Complexity quick-reference (balanced / typical)

- **RangeSet**: add/remove `O(log n + m)`, overlap/contains `O(log n)`.
    
- **Interval Tree**: insert/delete `O(log n)`, query `O(log n + k)`.
    
- **Fenwick**: `O(log n)` point/prefix (and the range tricks).
    
- **Segment Tree (lazy)**: `O(log n)` range update/query.
    
- **Difference Map**: updates `O(log n)` (ordered map); recompute metrics by scan `O(n)` unless you maintain a running prefix.
    

---

# 5) Gotchas you won’t trip over

- **Semantics**: use **half-open** `[L, R)`; explicitly decide if **adjacent** `[a,b)` + `[b,c)` should merge.
    
- **Precision**: for floats/timestamps, prefer integer ticks or careful comparison (EPS).
    
- **Compression**: when keys are large/sparse (timestamps, IPv4), coordinate-compress before Fenwick/Segment Tree.
    
- **Deletions**: DSU/Union-Find is **not** for deletions or overlap queries—only for _connectivity_ (clustering) if you already have overlaps.
    

---

# 6) Minimal “starter kits” (Python, already battle-tested)

- **RangeSet**: maintain union + overlaps/contains/length (you have the code above).
    
- **Interval Tree**: augmented BST with `max_end` (use earlier snippet; wrap in a balanced tree if worst-case insert order matters).
    
- **Fenwick**: prefix sums with compression for stabbing counts.
    
- **Segment Tree (lazy)**: range-add + range-sum (swap aggregator for min/max).
    

If you tell me your exact op mix (e.g., _“~200k intervals, heavy inserts, need `any-overlap([L,R))` and union length; occasional list-all overlaps”_), I’ll pin a final combo (e.g., **RangeSet + on-demand sweep**, or **Interval Tree + running union meter**) and hand you trimmed code for your language.

Here’s a concise, **developer-oriented rundown** of the main data structures for handling **ranges / intervals**, with tiny examples you can grasp at a glance:

---

## 🟩 **1. RangeSet (a.k.a. Interval Union)**

**Purpose:** Maintain the _union_ of intervals — disjoint, sorted ranges with fast add/remove and overlap checks.  
**Think:** “Covered or not?”

**Ops:**

- `add([L,R))`, `remove([L,R))`, `overlaps([L,R))`, `contains(x)`
    
- Time ≈ `O(log n + m)`
    

**Example:**

```python
S = RangeSet()
S.add(1,5)
S.add(3,7)   # merges → [1,7)
S.add(10,12)
S.remove(4,11)  # -> [(1,4), (11,12)]
```

✅ Fast overlap checks  
🚫 Doesn’t tell _which_ original intervals overlapped

---

## 🟦 **2. Interval Tree**

**Purpose:** Find all intervals that overlap a query `[L,R)`.  
**Think:** “Which intervals touch this one?”

**Ops:**

- Insert/Delete interval
    
- Query overlaps `[L,R)` → `O(log n + k)` (`k` = matches)
    

**Example:**

```python
T = IntervalTree()
T.insert(1,5)
T.insert(10,15)
T.insert(3,7)
T.search_all_overlaps(4,11)
# -> [(1,5), (3,7), (10,15)]
```

✅ Returns actual overlapping intervals  
🚫 Slightly heavier memory and code complexity

---

## 🟨 **3. Segment Tree**

**Purpose:** Range updates & queries on a **dense / numeric** axis.  
**Think:** “Sum/min/max/count over any range.”

**Ops:**

- `update(L,R,val)`
    
- `query(L,R)`
    
- Time ≈ `O(log n)` per op
    

**Example:**

```python
# Range add, range sum
update(2,5,+3)
query(0,5)  # sum of array[0..5)
```

✅ Powerful for aggregates  
🚫 Needs discrete numeric domain (arrays or compressed coords)

---

## 🟥 **4. Fenwick Tree (Binary Indexed Tree)**

**Purpose:** Simpler segment tree variant for prefix sums / point updates.  
**Think:** “Dynamic prefix sums or counts.”

**Ops:**

- `add(idx, delta)`
    
- `prefix(idx)`
    
- Time ≈ `O(log n)`
    

**Example:**

```python
add(2,+1)
add(5,-1)
prefix(4)  # -> active count at point 4
```

✅ Tiny and fast  
🚫 Limited to numeric indexes, one-dimensional aggregates

---

## 🟧 **5. Difference Map**

**Purpose:** Track overlap _counts_ by marking interval starts and ends (`+1`, `-1`).  
**Think:** “Sweep-line for dynamic updates.”

**Ops:**

- `delta[L]+=1`, `delta[R]-=1`
    
- Scan or prefix sum → active overlaps
    

**Example:**

```python
d = {}
d[1]=d.get(1,0)+1; d[5]=d.get(5,0)-1
d[3]=d.get(3,0)+1; d[7]=d.get(7,0)-1
# prefix scan: 1→1, 3→2, 5→1, 7→0
```

✅ Simple + great for batch overlap stats  
🚫 No direct random-access queries unless wrapped in a BIT

---

### 🔧 Quick summary

| Structure          | Best for                     | Dynamic? | Returns           | Typical Ops           | Complexity     |
| ------------------ | ---------------------------- | -------- | ----------------- | --------------------- | -------------- |
| **RangeSet**       | union / coverage             | ✅        | boolean / union   | add, remove, overlaps | `O(log n)`     |
| **Interval Tree**  | find overlapping intervals   | ✅        | list of intervals | insert, query         | `O(log n + k)` |
| **Segment Tree**   | numeric range aggregates     | ✅        | scalar            | update, query         | `O(log n)`     |
| **Fenwick Tree**   | prefix sums / counts         | ✅        | scalar            | add, prefix           | `O(log n)`     |
| **Difference Map** | quick overlap counts (batch) | ✅        | counts            | +1/-1 at ends         | `O(log n)`     |


### 🟩 **RangeSet**

- **What it does:** keeps the _union_ of disjoint, sorted intervals.
    
- **Use when:** you just need to know “is this covered?”, “does anything overlap?”, or “what’s the total covered length?”.
    
- **Example:**
    
    - add `[1,5)`, add `[3,8)` → merges into `[1,8)`
        
    - remove `[4,6)` → results in `[1,4)` and `[6,8)`
        

---

### 🟦 **Interval Tree**

- **What it does:** stores original intervals in a tree, augmented with each node’s `max_end`.
    
- **Use when:** you must _find all intervals that overlap_ a given `[L,R)`.
    
- **Example:**
    
    - intervals `(1,5),(3,7),(10,15)`
        
    - query `[4,11)` → overlaps are `(1,5),(3,7),(10,15)`.
        

---

### 🟨 **Segment Tree**

- **What it does:** supports **range updates** and **range queries** over a _dense numeric_ axis.
    
- **Use when:** you need sums, mins, or max over arbitrary subranges (e.g. histogram updates).
    
- **Example:**
    
    - update `[2,5)` by +3
        
    - query `[0,5)` → sum = total in that range.
        

---

### 🟥 **Fenwick Tree** (Binary Indexed Tree)

- **What it does:** compact version of segment tree for **prefix sums / counts**.
    
- **Use when:** you only need one direction of aggregation (e.g. prefix sum).
    
- **Example:**
    
    - add +1 at index 2, -1 at index 5
        
    - prefix(4) → count of active intervals at point 4.
        

---

### 🟧 **Difference Map**

- **What it does:** tracks deltas `+1` at start, `-1` at end; prefix gives overlap count.
    
- **Use when:** batch processing or simple dynamic overlap counting.
    
- **Example:**
    
    - `[1,5)` → `+1@1, -1@5`
        
    - `[3,7)` → `+1@3, -1@7`
        
    - prefix scan → `1→1, 3→2, 5→1, 7→0`.
        

---

### 🟪 **RangeMap**

- **What it does:** like RangeSet but each segment stores a **value/label** (piecewise constant function).
    
- **Use when:** intervals carry a value (like access level, cost, ACL).
    
- **Example:**
    
    - `[0,10)=A`, `[10,20)=B`; assign `[15,25)=C` → merges and splits as needed.
        

---

