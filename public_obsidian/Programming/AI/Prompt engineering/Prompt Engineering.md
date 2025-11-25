

#    TACTICS template (Task - Actor - Context - Target - Input - Constraints - Specimen)

Let's say you want Claude to help you create a recipe. Here's the **correct** TACTICS breakdown:

## 🎯 **T - Task**

_What you want done_ "Create a chocolate chip cookie recipe"

## 👤 **A - Actor**

_Who Claude should act as_ "Act as an experienced pastry chef teaching a cooking class"

## 🌍 **C - Context**

_Background information_ "I'm a beginner baker with a basic kitchen. I'm making these for my kid's bake sale tomorrow and have never baked from scratch before."

## 🎪 **T - Target**

_Who the output is for_ "The recipe is for beginners who might feel intimidated by baking"

## 📥 **I - Input**

_What you're providing_ "I have: flour, sugar, butter, eggs, chocolate chips, vanilla, baking soda, and salt"

## 🚧 **C - Constraints**

_Limitations or requirements_ "Must take under 30 minutes total, make exactly 24 cookies, and use only one mixing bowl"

## 📋 **S - Specimen**

_Example of desired output format_ "Format like this: **Prep Time:** X minutes **Ingredients:** (bulleted list) **Steps:** (numbered, with why each step matters)"

# Challenge 1: Extract IOCs from Threat Report (Text → JSON)

**Input: Messy Threat Intelligence Text**

|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `APT28 campaign detected targeting defense contractors. Observed C2 servers at 185.220.101.45`<br><br>`and 162.55.90.122 (both on port 443). Malware drops files with SHA256`<br><br>`a3b4c5d6e7f8g9h0i1j2k3l4m5n6o7p8q9r0s1t2u3v4w5x6y7z8a9b0c1d2e3f4 and`<br><br>`communicates to evil-domain.com and malware-c2.net. Also seen IPs 10.0.0.1 (internal)`<br><br>`and 192.168.1.100 (local). Email addresses involved: attacker@evil.com, victim@defense.gov.`<br><br>`CVE-2024-1234 and CVE-2024-5678 exploited. Attack uses PowerShell and cmd.exe.` |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |

used chatgpt to craft prompt
You will use the TACTICS pattern to convert unstructured cyber threat text into structured IOC JSON.

T – Task  
Convert unstructured cyber threat / incident report text into a strict, machine-readable IOC JSON object. Extract indicators of compromise (IOCs) and related context, and normalize them into a consistent schema.

A – Actor  
Act as a senior SOC analyst and DFIR engineer who specializes in:
- Parsing messy threat intel reports
- Identifying and classifying IOCs
- Normalizing them into a clean JSON schema similar to STIX-style structures
You are meticulous about data types, IP classifications, and never invent information that isn’t in the input.

C – Context  
I’m building an automated pipeline that ingests free-text threat intel (emails, reports, analyst notes) and turns them into structured IOCs.
The JSON you produce will be:
- Ingested by security tools (SIEM/SOAR)
- Reviewed by junior analysts
So it must be clear, consistent, and strictly valid JSON.

T – Target  
The output is for:
- Automated systems that require valid JSON only
- Security analysts who want a quick IOC set without reading the full narrative

I – Input  
I will provide you with unstructured threat text like this (example):

"APT28 campaign detected targeting defense contractors. Observed C2 servers at 185.220.101.45
and 162.55.90.122 (both on port 443). Malware drops files with SHA256
a3b4c5d6e7f8g9h0i1j2k3l4m5n6o7p8q9r0s1t2u3v4w5x6y7z8a9b0c1d2e3f4 and
communicates to evil-domain.com and malware-c2.net. Also seen IPs 10.0.0.1 (internal)
and 192.168.1.100 (local). Email addresses involved: attacker@evil.com, victim@defense.gov.
CVE-2024-1234 and CVE-2024-5678 exploited. Attack uses PowerShell and cmd.exe."

From such text, extract at least:
- Campaign and actor (campaign name, threat actor, short description)
- IP addresses (with classification: public vs private/internal/local, role, ports if mentioned)
- Domains / hostnames
- File hashes (with algorithm type)
- Email addresses (with role if inferable: attacker / victim / unknown)
- Exploited vulnerabilities (CVEs)
- Tools / techniques (e.g., PowerShell, cmd.exe)

C – Constraints  
1. Output ONLY JSON. No prose, no markdown, no explanation.
2. JSON must be syntactically valid and parseable.
3. Use this schema (omit fields that don’t exist in the input; do NOT invent values):

{
  "campaign": {
    "name": null,
    "threat_actor": null,
    "description": null
  },
  "indicators": {
    "ipv4": [
      {
        "ip": "string",
        "port": 0,
        "role": "c2 | internal | local | unknown",
        "is_private": false
      }
    ],
    "domains": [
      {
        "domain": "string",
        "role": "c2 | phishing | unknown"
      }
    ],
    "file_hashes": [
      {
        "hash": "string",
        "algorithm": "sha256 | md5 | sha1 | unknown",
        "role": "malware_binary | dropped_file | unknown"
      }
    ],
    "emails": [
      {
        "address": "string",
        "role": "attacker | victim | unknown"
      }
    ],
    "vulnerabilities": [
      {
        "cve": "string"
      }
    ],
    "tools": [
      {
        "name": "string",
        "type": "living_off_the_land | malware | unknown"
      }
    ]
  }
}

Classification rules:
- RFC1918/private ranges → "is_private": true; role "internal" or "local" if stated.
- Public IPs → "is_private": false; if described as C2, role "c2".
- If a value is clearly mentioned, fill it in. If unknown, use "unknown" or null as appropriate.
- Use arrays for multiple values of the same type (IPs, domains, hashes, emails, etc.).
- Do not add any extra top-level keys beyond "campaign" and "indicators".

S – Specimen  
For the example input text above, your JSON SHOULD look structurally like this (values must match the actual input):

{
  "campaign": {
    "name": "APT28 campaign targeting defense contractors",
    "threat_actor": "APT28",
    "description": "APT28 campaign detected targeting defense contractors."
  },
  "indicators": {
    "ipv4": [
      {
        "ip": "185.220.101.45",
        "port": 443,
        "role": "c2",
        "is_private": false
      },
      {
        "ip": "162.55.90.122",
        "port": 443,
        "role": "c2",
        "is_private": false
      },
      {
        "ip": "10.0.0.1",
        "port": null,
        "role": "internal",
        "is_private": true
      },
      {
        "ip": "192.168.1.100",
        "port": null,
        "role": "local",
        "is_private": true
      }
    ],
    "domains": [
      {
        "domain": "evil-domain.com",
        "role": "c2"
      },
      {
        "domain": "malware-c2.net",
        "role": "c2"
      }
    ],
    "file_hashes": [
      {
        "hash": "a3b4c5d6e7f8g9h0i1j2k3l4m5n6o7p8q9r0s1t2u3v4w5x6y7z8a9b0c1d2e3f4",
        "algorithm": "sha256",
        "role": "malware_binary"
      }
    ],
    "emails": [
      {
        "address": "attacker@evil.com",
        "role": "attacker"
      },
      {
        "address": "victim@defense.gov",
        "role": "victim"
      }
    ],
    "vulnerabilities": [
      {
        "cve": "CVE-2024-1234"
      },
      {
        "cve": "CVE-2024-5678"
      }
    ],
    "tools": [
      {
        "name": "PowerShell",
        "type": "living_off_the_land"
      },
      {
        "name": "cmd.exe",
        "type": "living_off_the_land"
      }
    ]
  }
}

Now, I will provide the actual threat text. Return ONLY the JSON as specified above.

# 