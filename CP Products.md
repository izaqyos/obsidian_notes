Check point products

- [Harmony SASE](#Harmony%20SASE)
- [Harmony SASE](#Harmony%20SASE)
	- [Private Access](#Private%20Access)
	- [](#)


![[Pasted image 20251001131049.png]]

links:
https://www.checkpoint.com/cyber-hub/
https://wiki.checkpoint.com/confluence/pages/viewpage.action?spaceKey=SALESENAB&title=Product+Enablement+Wiki
https://catalog.checkpoint.com/

## Harmony SASE
![[Pasted image 20251001140839.png]]


![[Pasted image 20251005181221.png]]


![[Pasted image 20251005182714.png]]

![[Pasted image 20251005182841.png]]

### Private Access
![[Pasted image 20251005183141.png]]
![[Pasted image 20251005183529.png]]
### Architecture
![[Pasted image 20251006123238.png]]

### resources
![[Pasted image 20251006124554.png]]

### Next Gen Arch
![[Pasted image 20251006125338.png]]

### GW

#### GW HL
![[Pasted image 20251006133752.png]]
![[Pasted image 20251006134001.png]]
missing is firefly - secure access to on prem resources over wireguard / ipsec 
Nomad <-> Nomad agent , k8s - kubectl like , containers mgmt 
SDP - Guard , Secure GW controller communicated to different services via guard 
consul - service discovery
SXDNS - API interface to consul
SDPC - interacts with consul via SX DNS and serves the network access requests from agents. for establishing connection it sends configuration + GW back to agent

#### ZTA
![[Pasted image 20251006135018.png]]
ZTA config UX
![[Pasted image 20251006135201.png]]

ZTA access
![[Pasted image 20251006135329.png]]

ZTA components
![[Pasted image 20251006135430.png]]
### wireguard

#### desc
level 3 (IP) protocol, UDP 
WireGuard is a modern, minimal L3 VPN that runs over UDP and uses a fixed, opinionated crypto suite: Curve25519 (key exchange), ChaCha20-Poly1305 (AEAD), BLAKE2s (hash), and HKDF (key derivation). It uses a NoiseIK-based handshake with perfect forward secrecy, typically completes in 1 RTT, and can add an optional pre-shared key. Peers are identified by static public keys and routing is defined via `AllowedIPs`. It’s NAT-friendly (keepalives) with seamless roaming, has a tiny codebase (performance + small attack surface), and defaults to port 51820/UDP. Configuration is dead simple: one interface, a few peers, no cipher negotiation.


#### config
![[Pasted image 20251006135720.png]]


### IPSEC

multi tunnel support (for redundancy)
![[Pasted image 20251006135945.png]]

### Network deployment
saferx handler 

![[Pasted image 20251009120231.png]]

### Connect to Network
![[Pasted image 20251009120306.png]]

### S