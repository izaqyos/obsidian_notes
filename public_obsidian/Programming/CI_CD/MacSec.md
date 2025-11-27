




---

## 1. What MACsec actually secures

**MACsec = IEEE 802.1AE**: it’s a standard for **link-layer (Layer 2) encryption and integrity** on Ethernet links.

On a MACsec-protected link:

- Each Ethernet frame is wrapped in a **MACsec header + trailer**:
    
    - **SecTAG** (security tag) goes after the MAC addresses.
        
    - **ICV** (Integrity Check Value) is appended at the end.
        
- It uses **AES-GCM** (Galois/Counter Mode) for **authenticated encryption**:
    
    - Provides both **confidentiality** (encryption) and **integrity + authenticity** in one shot.
        
- What stays visible:
    
    - **Source/destination MAC addresses**
        
    - Some MACsec metadata (SCI can be implicit or explicit)
        
- What gets protected:
    
    - Higher-layer headers (IP, TCP, etc.)
        
    - Most of the original Ethernet payload
        
    - Integrity over almost the entire frame
        

**Hop-by-hop model:**  
MACsec works **on each physical link**. A frame is decrypted when it arrives at a switch/router, processed, and then **re-encrypted** when it’s sent out on another MACsec-enabled port. This is different from end-to-end IPsec/TLS.

---

## 2. MACsec building blocks (concepts)

To talk usage, a few terms:

- **CA (Connectivity Association)**  
    A group of MACsec participants that share a common policy and keys. Usually: “these two ports” or “this set of ports” on a link.
    
- **Secure Channel (SC)**  
    A **unidirectional** logical channel bound to a transmitter (identified by an SCI – Secure Channel Identifier).  
    **Each transmitter has its own SC** for outbound traffic.
    
- **Secure Association (SA)**  
    A particular **key instance** within a Secure Channel. Each SC can have multiple SAs (for key rotation). Each SA is associated with:
    
    - A **Secure Association Key (SAK)**
        
    - A **Packet Number (PN)** counter
        
    - Crypto parameters (cipher suite, etc.)
        
- **SCI (Secure Channel Identifier)**  
    Usually **{system MAC, port ID}** – globally identifies the transmitter side of a MACsec link. Can be sent explicitly in the SecTAG or inferred.
    

The high-level flow:

1. Devices agree on **which CA they’re in**.
    
2. Within the CA, each transmit port sets up an **SC**.
    
3. Within each SC, they establish **SAs** (keys/parameters).
    
4. Frames are sent and received only if they match an active SA with valid PN, ICV, etc.

# diagrams  


- Where MACsec sits in the stack
    
- Typical **hop-by-hop use** in the network
    
- **Key management (802.1X + MKA)** flow
    
- **Frame transformation** on a MACsec link
    

---

### 1) Where MACsec sits (layering)

```text
+-------------------------------+
|      Application (TLS)        |
+-------------------------------+
|    Transport (TCP / UDP)      |
+-------------------------------+
|      Network (IP / IPsec)     |
+-------------------------------+
|   Data Link (Ethernet +       |
|          MACsec)              |
+-------------------------------+
|     Physical (Fiber/Copper)   |
+-------------------------------+
```

- MACsec lives at **Layer 2** (Ethernet).
    
- It can coexist with **IPsec** (L3) and **TLS** (L7) as extra layers of defense.
    

---

### 2) Hop-by-hop MACsec in a network

```text
        [Host A]
           |
           |  (MACsec link: encrypted L2)
           v
   +------------------+
   |  Access Switch   |
   | (MACsec on port) |
   +--------+---------+
            |
            |  (MACsec link: encrypted L2)
            v
   +------------------+
   |   Distribution   |
   |      Switch      |
   +--------+---------+
            |
            |  (MACsec link: encrypted L2)
            v
   +------------------+
   |      Core        |
   +------------------+
            |
            |  (WAN / DCI, often MACsec too)
            v
        [Remote Site]
```

- Each **physical L2 link** can be MACsec-protected.
    
- Traffic is **decrypted inside each switch/router**, processed, then **re-encrypted** on the next link.
    

---

### 3) Key management: 802.1X + MKA + MACsec

High-level control-plane vs data-plane view:

```text
      Control Plane (Auth & Key Management)
      -------------------------------------
          +------------------------------+
          |      Authentication Server   |
          |       (RADIUS / AAA)         |
          +--------------^---------------+
                         |
                         | RADIUS (EAP, CAK delivery, policies)
                         |
+------------------------+------------------------+
|          Access Switch (Authenticator)          |
|                                                |
|  802.1X + MKA session with Host A on this port |
+------------------------^-----------------------+
                         |
                         | EAPOL (802.1X) / MKA
                         |
                  [Host A (Supplicant)]


      Data Plane (MACsec-encrypted link)
      ----------------------------------
      [Host A] ===<===<===<===>=== MACsec ===>===<===<=== [Access Switch]
                      (AES-GCM on each frame)
```

Flow in words:

1. Host A authenticates via **802.1X (EAPOL)** to the **Access Switch**.
    
2. Switch talks to **RADIUS**; they agree on a **CAK** and MACsec policy.
    
3. **MKA** (MACsec Key Agreement) runs between Host A and switch, derives **SAKs**.
    
4. Data-plane traffic on that port is then **encrypted with MACsec**.
    

---

### 4) Connectivity Association / Secure Channel / SAs (conceptual)

For a single MACsec-protected link:

```text
              Connectivity Association (CA)
        -----------------------------------------
        Members: Host A port, Switch port
        Policy: cipher, replay window, etc.
        -----------------------------------------

   Transmit side (Host A)                 Receive side (Switch)
   ------------------------               -----------------------
   Secure Channel (SC)   ---->           Secure Channel (SC)
        |                                      |
        +-- Secure Assoc #1 (SA1, old key)     +-- SA1 (rx only)
        |                                      |
        +-- Secure Assoc #2 (SA2, active)      +-- SA2 (rx + accept)
              (Key = SAK2, PN counter)              (Key = SAK2)
```

- Each **transmitter** has an **SC** (unidirectional).
    
- Each SC has **one or more SAs** (for rekeying).
    
- MKA rotates SAs over time.
    

---

### 5) MACsec frame transformation (per packet)

Original frame:

```text
+------------+------------+-----------+-----------+
| Dest MAC   | Src MAC    | EtherType |  Payload  |
+------------+------------+-----------+-----------+
```

MACsec-protected on the wire:

```text
+------------+------------+-----------+----------+----------+
| Dest MAC   | Src MAC    | 0x88E5    | SecTAG   | Encrypted|
|            |            | (MACsec)  | (SCI, PN)| Payload  |
+------------+------------+-----------+----------+----------+
                                                    +------+
                                                    | ICV  |
                                                    +------+
```

- **SecTAG**: includes things like SCI (ID of the transmitter), PN (packet number).
    
- **ICV**: integrity check value from AES-GCM.
    

---

### 6) Typical use-case topologies (HL)

#### a) Campus access

```text
[User PC] --MACsec--> [Access Switch] --MACsec--> [Core]
        (copper)                   (uplink fiber)
```

- Protects the **local wire** against taps.
    
- Often combined with **802.1X port-based access control**.
    

#### b) Spine-leaf in DC

```text
        +--------- Spine 1 ---------+
        |                            |
   MACsec links                  MACsec links
        |                            |
      Leaf 1                      Leaf 2
        |                            |
     Servers                      Servers
```

- All east-west links are **encrypted at L2**.
    
- Transparent to VXLAN/IP fabrics above.
    

#### c) DCI / WAN over Ethernet

```text
        DC1                                      DC2
+----------------+                        +----------------+
|  DC1 Core      |==== MACsec L2 link ====|  DC2 Core      |
+----------------+      over provider     +----------------+
                 \                        /
                  \____ Untrusted L2 ____/
```

- MACsec runs **over** the provider’s Layer-2 service to hide your traffic.
    

---

If you tell me where you want to put these (design doc, training slide, runbook), I can tweak the diagrams for that context (e.g., one-page “MACsec at a glance” slide, or a more detailed “control-plane vs data-plane” diagram).

---

## 3. Key management: 802.1X, MKA, and SAKs

MACsec encryption itself (802.1AE) is separate from **key management**. Keys are distributed by **MACsec Key Agreement (MKA)**, defined in **IEEE 802.1X-2010**.

### 3.1 Roles and components

Typically you have:

- **Supplicant** – a MACsec-capable host or switch port that wants access.
    
- **Authenticator** – usually a switch, enforcing port access and MACsec policies.
    
- **Authentication Server** – usually a RADIUS server (e.g., Cisco ISE, ClearPass, etc.) validating credentials and pushing MACsec policies.
    

Process:

1. **802.1X authentication**
    
    - EAP over LAN (EAPOL) between Supplicant & Authenticator.
        
    - Authenticator talks RADIUS to the Authentication Server.
        
    - If authentication succeeds, everyone derives a **CAK (Connectivity Association Key)** or gets it from RADIUS attributes.
        
2. **MKA session (Key Agreement)**
    
    - Uses the CAK to protect **MKA messages**.
        
    - Establishes the **Connectivity Association (CA)**.
        
    - Negotiates:
        
        - Who is in the CA (peers, their SCI, capabilities).
            
        - Which cipher suites and policies to use (e.g. AES-128-GCM or AES-256-GCM).
            
    - Derives and distributes one or more **SAKs (Secure Association Keys)** that actually encrypt the traffic.
        
3. **Key rotation & rekeying**
    
    - MKA rotates SAKs periodically or based on PN thresholds/time.
        
    - Multiple SAs per SC allow **hitless rekeying**:
        
        - While SA#1 is active, SA#2 is pre-installed.
            
        - At a coordinated time, sender flips to SA#2; receiver accepts both for a grace period.
            

### 3.2 Policies you typically control

As a network engineer, you’ll configure (vendor syntax differs):

- **Cipher suite:** AES-128-GCM vs AES-256-GCM.
    
- **Replay protection window** (e.g. 0, 32, 1024 packets).
    
- **MKA lifetime / rekey interval.**
    
- **Whether to allow “clear-text” fallback** (unsecured) or “must-secure.”
    

---

## 4. Packet flow: what happens to a frame

For an **outgoing frame** on a MACsec-enabled port:

1. Original Ethernet frame is created (Dst MAC, Src MAC, EtherType, payload).
    
2. MACsec engine:
    
    - Looks up the **active transmit SA** for this SC.
        
    - Increments **PN** and inserts it in the MACsec SecTAG.
        
    - Encrypts payload with AES-GCM using SAK + PN + SCI as inputs.
        
    - Generates the **ICV** (integrity tag) and appends it.
        
3. Frame goes on the wire as a **MACsec frame** (Ethertype `0x88E5`).
    

For an **incoming frame**:

1. Port receives frame with EtherType 0x88E5 (MACsec).
    
2. MACsec engine:
    
    - Uses the SCI (explicit or inferred) to find the right **SC**.
        
    - Uses PN to pick the right **SA** (and checks replay window).
        
    - Verifies the ICV (integrity/authenticity).
        
    - If valid, decrypts payload and reconstructs original Ethernet frame.
        
3. The switch/router now sees a “normal” Ethernet frame and forwards it based on regular L2/L3 logic.
    

If any of the checks fail (wrong key, PN out of window, corrupted data), the frame is **discarded**.

---

## 5. Where MACsec is used in real networks

### 5.1 Access layer / campus

Use cases:

- Encrypting **user → access switch** links (wired 802.1X).
    
- Protecting **printer / IP phone / endpoint** traffic on the local copper link from office “taps.”
    
- Corporate environments where **“encrypt everywhere”** is a requirement.
    

Pros:

- Transparent to higher layers (no IPsec configuration on hosts).
    
- Very low added latency; typically line-rate in silicon.
    

### 5.2 Data center interconnect (DCI) and spine-leaf

Use cases:

- Encrypting **switch-to-switch** links (spine-leaf, ToR-spine).
    
- Securing **DCI over dark fiber / wave services** (sometimes combined with optical encryption).
    

Pros:

- Simple: configure on both ends of the link; no overlay awareness.
    
- Good for **east-west** protection (intra-DC) where IPsec/TLS may not be practical.
    

### 5.3 WAN over Ethernet / Carrier services

Where you have:

- **Layer 2 VPNs (EPL, EVPL, E-LAN)** from a provider.
    
- Need to protect against the provider seeing or tampering with traffic.
    

MACsec is often deployed **CE-to-PE** or **CE-to-CE** over provider Ethernet.

---

## 6. MACsec vs other security options

### 6.1 MACsec vs IPsec

- **Where they act:**
    
    - MACsec: **L2**, per-link, local segment only.
        
    - IPsec: **L3**, end-to-end or site-to-site across routed networks.
        
- **Topology dependence:**
    
    - MACsec: every hop that should be protected needs MACsec.
        
    - IPsec: can tunnel through an untrusted core; only endpoints need IPsec.
        
- **Operational complexity:**
    
    - MACsec: simple but many links to configure.
        
    - IPsec: fewer endpoints but more complex routing/tunnel configuration.
        
- **Overhead:**
    
    - MACsec: fixed per-frame overhead; very low CPU use if in hardware.
        
    - IPsec: MTU adjustments, tunnels, potential CPU impact if not offloaded.
        

### 6.2 MACsec vs TLS

- TLS is **application-level**.
    
- MACsec protects **all traffic on a link** (including protocols that don’t do TLS).
    
- In many designs you use **both**: TLS for app-level E2E, MACsec to ensure the transport link is secure.
    

---

## 7. Practical design & configuration tips

Here’s what you usually think about when deploying MACsec.

### 7.1 Decide scope

- Do you encrypt:
    
    - Only **DCI links**?
        
    - All **spine–leaf links**?
        
    - Every **user access port**?
        
- More scope = more security but more:
    
    - Hardware requirements (all devices must support MACsec on the relevant ports).
        
    - Operational overhead.
        

### 7.2 Hardware/feature checks

- Verify:
    
    - Port types (some support MACsec only on specific interfaces/line cards).
        
    - Max throughput with MACsec (older gear may have limits).
        
    - Cipher suites supported (AES-128 vs AES-256).
        

Some vendors also support **“MACsec over L3”** or variants that interact with VXLAN/EVPN—details are vendor-specific.

### 7.3 Authentication & key management design

- Decide:
    
    - **Centralized 802.1X/MKA with RADIUS** vs static pre-shared keys (the latter is rare in serious deployments).
        
    - RADIUS attributes that control:
        
        - Whether a port must use MACsec.
            
        - Cipher suite.
            
        - Whether clear traffic is allowed if MACsec negotiation fails.
            
- For high-security:
    
    - **Disable fallback to clear-text**.
        
    - Use **replay protection window = 0 or very small** (at some cost in tolerance to reordering).
        

### 7.4 MTU and overhead

MACsec adds **header + ICV bytes**, effectively increasing frame size. You may need to:

- Increase **port MTU** on MACsec links.
    
- Ensure paths in the provider network can carry the slightly larger frames (especially if using QinQ, VXLAN, etc.).
    

### 7.5 Monitoring & troubleshooting

Typical things to watch:

- **MACsec session status** (up/down per port).
    
- **MKA session state** (auth, keys installed).
    
- Counters:
    
    - Valid/invalid frames.
        
    - Replay drops.
        
    - “Not using MACsec” frames if you allow mixed traffic.
        

When something breaks:

- Check 802.1X/EAP logs (auth failure).
    
- Check RADIUS policy (maybe not pushing MACsec attributes).
    
- Check that both sides agree on cipher suites and have matching CAK/SAKs.
    

---

## 8. When should you _not_ rely only on MACsec?

MACsec **does not**:

- Provide end-to-end crypto across multiple hops/ISP networks.
    
- Protect traffic once it leaves a MACsec-enabled domain.
    
- Replace host-level or app-level security.
    

If your threat model includes:

- Compromise of **core routers**, not just physical taps.
    
- Untrusted intermediate ISPs.
    
- Malicious servers/endpoints.
    

…then you still need **IPsec, TLS, SSH, etc.** on top of MACsec. MACsec is strongest as a **defense-in-depth layer** for physical/tap risk and “local wire” security.

---

If you tell me roughly **what environment you’re thinking about** (e.g., “campus with 802.1X already,” “DC spine-leaf,” “MPLS L2VPN over provider X”, etc.), I can walk through:

- A concrete **design pattern** for MACsec there, and
    
- Example **config snippets** (generic or even vendor-specific like Cisco/Nokia/Juniper/Aruba) showing how you’d actually enable it and integrate with RADIUS.