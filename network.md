# Network

### Layers
| Layer | Common Protocols | Encapsulated Data | Address Type | Managed by | Directed by |
|---|---|---|---|---|---|
| Application | HTTP, HTTPS, FTP, SSH, SMTP, POP3, IMAP, DNS | Messages | N/A | Application software (e.g., browsers, email clients) | N/A |
| Transport | TCP, UDP | Segments (TCP) or Datagrams (UDP) | Port Number | OS | N/A |
| Internet | IP | Packets | IP Address | OS | Routers |
| Network Access | Ethernet, WiFi, DSL, ISDN | Frames | MAC Address | NIC | Switches, Access Points |

- SMTP (Simple Mail Transfer Protocol) sends email. POP3 (Post Office Protocol version 3) and IMAP (Internet Message Access Protocol) retrieves email. POP3 deletes email from the server upon retrieval, whereas IMAP doesn't.
- DSL (Digital Subscriber Line) and ISDN (Integrated Services Digital Network) provides digital data transmission over the wires of a telephone network.

## Router
The internet is a global network of routers.

### Switch
- Switch: Operates at the Network Access layer. Decapsulates the data to verify the MAC address before passing it along within the same network.
- Router: Operates up to the Internet layer. Decapsulates the data to verify the IP address before forwarding it to a different network.

### Modem, Access Point, Gateway
- Modem: An analog (telephone lines or cable) <-> digital converter.
- Router: Connected to a modem and handles DHCP (Dynamic Host Configuration Protocol) and NAT (Network Address Translation).
- Access Point (AP): Broadcasts WiFi signals.
- Gateway: A single device combines functionalities of modem, router, and access point. Or, in a broader context of networking, it refers to a network node that acts as an entry point from one network to another.

### traceroute and ping
- traceroute:
  - E.g., `traceroute github.com`
  - Returns RTT (Round Trip Time) for each hop to routers.
  - `* * *` indicates that a router isn't configured to return the ICMP (Internet Control Message Protocol) echo request messages.
  - TTL (Time To Live) represents the maximum number of hops to try to reach for the final destination.
  - It is used for detecting problematic routers in the network. Such routers return high (200 ms or so) RTTs.
- ping:
  - E.g., `ping github.com`
  - Similar to traceroute, but provides the RTT for the final destination router only.

## Public vs Private IP Address
- Public IP address: Assigned by an ISP for external use.
- Private IP address: Assigned by a router for internal use.
- To find your IP address:
  - Public: `curl ifconfig.me`
  - Private: `ifconfig`
- Both public and private IP addresses can change periodically.
- Shortage (256^4 ~= 4B) of IPv4 addresses led to the development of private IP addresses.
- Private IP address has to be converted into public IP address in order to access the internet.
- Within your router, DHCP assigns your internal devices a private IP address, and NAT translates between private and public IP addresses.

### Subnet Mask
- An IP address consists of two parts: network address and host address. 
- A subnet mask masks the network portion of the IP address, with 1s indicating the network portion.
- For example, a private IP address 192.168.0.2 has a subnet mask 255.255.255.0, indicating that the first three octets are the network portion.
- If it shares the network portion, it's in the same subnet.
- Networks are broken down into subnets to reduce ARP broadcasting.
- `ifconfig` gives the IP address of the computer, subnet mask, and default gateway.
- The host address with all 1s is used for broadcasting, while the one with all 0s is the network address.

### CIDR (Classless Inter-Domain Routing) Notation
| IP Address Range | CIDR Notation |
|---|---|
| 1.0.0.0 - 1.255.255.255 | 1.0.0.0/8 |
| 1.30.0.0 - 1.30.255.255 | 1.30.0.0/16 |
| 1.30.128.0 - 1.30.255.255 | 1.30.128.0/17 |


### Public IP Address Ownerships
The ICANN (Internet Corporation for Assigned Names and Numbers) and RIRs (Regional Internet Registries) manage public IP address distribution:

| RIR | Region | Address count | % of total |
|---|---|---|---|
| ARIN | North America | 1,662,226,432 | 38.7% |
| APNIC | Asia-Pacific | 887,005,696 | 20.7% |
| RIPE NCC | Europe, Central Asia, and West Asia | 831,209,096 | 19.4% |
| LACNIC | Latin America and Caribbean | 189,931,520 | 4.4% |
| AFRINIC | Africa | 115,739,648 | 2.7% |

- Apple owns 17.0.0.0/8, an example of a /8 IPv4 address block.
- US DoD owns fourteen /8 blocks.
- 10.0.0.0/8 block is reserved for private IP addresses.
- 127.0.0.0/8 is reserved for loopback addresses.

### Private IP Address Classes
| Class | IP Address Range | Default Subnet Mask | Suited For |
|---|---|---|---|
| Class A | 10.0.0.0 - 10.255.255.255 | 255.0.0.0 | Large organizations |
| Class B | 172.16.0.0 - 172.31.255.255 | 255.255.0.0 | Medium-sized organizations |
| Class C | 192.168.0.0 - 192.168.255.255 | 255.255.255.0 | Small organizations or homes |

### Loopback Addresses
- Loopback addresses are used for local machine communication.
- 127.0.0.0/8 is designated for loopback addresses.
- 127.0.0.1 is often referred to as 'localhost'.
- A computer's network interface recognizes this special address and loops any messages back to the same machine, which is useful for testing.

### Port Forwarding
- Port forwarding uses port numbers to specify a computer within a private network.
- Port numbers are always associated with an IP address; the IP address leads to a server, while the port number leads to a specific service (like web service) on that server.
- The OS manages network communication by assigning and managing port numbers for applications seeking to communicate over the network.
- The configuration page of a network router contains a port forwarding or NAT table. This table maintains the mapping between internal and external IP addresses and port numbers.
- `netstat` gives the private IP address with port number on a local device and corresponding foreign IP address with port number it's connected to.
- Port numbers range from 0 to 65535:
  - 0 - 1023: Well-known ports (e.g., 80, 443 for HTTP/HTTPS, 21 for FTP, 25 for SMTP)
  - 1024 - 49151: Registered ports (registered by companies, e.g., 1101 for Adobe server, 1433 for Microsoft SQL server)
  - 49152-65535: Dynamic or private ports (temporary ports used by clients)

### Firewall
- Firewalls block incoming or outgoing packets based on rules, including IP address, domain name, protocols, programs, ports, or keywords.
- Firewalls can be host-based (on a computer) or network-based (on a router).

## ARP (Address Resolution Protocol)
- ARP maps local IP addresses to their matching MAC addresses. 
- `arp -a` gives the ARP cache.
- Entries in the cache can be either dynamic (temporary) or static (manual).

### ARP in Local Network Context
- When data arrives at a destination router's NIC (Network Interface Card), the ARP, using the target IP address, determines the corresponding MAC (Media Access Control) address, a 6-byte hexadecimal value burned into every NIC.
- ARP initiates the process by setting the target MAC address as `FF:FF:FF:FF:FF:FF` and broadcasting this address. The computer with the target IP address responds with its MAC address.

### ARP in Larger Network Context
- For inter-network communication, the current router assesses the optimal route and fetches the MAC address of the subsequent router using ARP.
- This process continues until the data reaches the final destination.
- Thus, while the sender and destination IP addresses remain constant, the destination MAC address is updated at each hop during the data transmission process.

## DNS (Domain Name System)
- DNS is an application layer protocol.
- The DNS data is cached at various levels.
- This mechanism, known as Recursive DNS Lookup, involves checking the web browser cache, the OS cache, the resolver server, the root server, the TLD server, and finally, the authoritative name server.

## HTTP (Hypertext Transfer Protocol)
- HTTP Request consists of:
  - Request line: This includes the request method (GET, POST, PUT, PATCH, DELETE), the request URI (Uniform Resource Identifier), and the HTTP version. (e.g., GET /hello-world HTTP/1.1)
  - Request headers: These are additional details in key-value pairs that define the operating parameters of an HTTP transaction Headers can include information about the client browser, the requested page, the server, and more.
  - Reqeust body: This contains data that the client sends to the server. It's typically empty for GET requests, as these requests are intended to fetch data rather than send it. For requests that manipulate server resources (like POST, PUT, and PATCH), the request body carries the necessary data.
- HTTP Response consists of:
  - Status Line: This includes the HTTP version, a status code, and a status text (e.g., "HTTP/1.1 200 OK").
  - Response Headers: These are additional details, in key-value pairs, that define the operating parameters of an HTTP response.
  - Response Body: This is the part of the response message that contains the requested resource, which could be HTML, CSS, JavaScript, images, or data.
- After the server processes a request, it might generate dynamic resources, such as building an HTML page with user-specific data.
- When the client (usually a web browser) receives the HTTP response, it parses and renders the HTML. As it does this, the browser might issue additional HTTP requests to fetch embedded resources, such as JavaScript files, CSS files, images, and data necessary to fully render the webpage to the user.

### URL Schemes
- A URL scheme specifies the protocol to be used when accessing the resource in the URL. Examples include:
  - `http://www.example.com`
  - `ftp://ftp.example.com`
  - `file:///home/username/path/to/your/file.txt`
  - `mailto:example@example.com` (opens the default mail client)
  - `tel:+1234567890`
  - `sms:+1234567890`
  - `magnet:?xt=urn:btih:QHQXPYWMACKDWKP47RRVIV7VOURXFE5Q&dn=My+Example+File`
    - `xt`: Exact Topic
    - `urn`: Uniform Resource Name
    - `btih`: BitTorrent Info Hash
    - `dn`: Display Name
