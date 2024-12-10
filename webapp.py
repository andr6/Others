#!/usr/bin/env python3

import subprocess
import argparse
import requests
from urllib.parse import urlparse

def fix_url_scheme(url):
    supported_schemes = ['http', 'https']
    if any(url.startswith(scheme + '://') for scheme in supported_schemes):
        return url
    return 'http://' + url

def check_redirect(url):
    try:
        response = requests.get(url, allow_redirects=False)
        if response.status_code in [301, 302] and response.headers.get('Location', '').startswith('https'):
            return 'https://' + url.split('://', 1)[1]
    except requests.RequestException:
        pass
    return url

def run_command(command, output_file):
    result = subprocess.run(command, capture_output=True, text=True)
    output_file.write(f"Command: {' '.join(command)}\n")
    output_file.write(result.stdout)
    output_file.write(result.stderr)
    output_file.write("\n\n")

def identify_technologies(url, output_file):
    output_file.write("## Identifying Technologies\n")
    run_command(["whatweb", "-a", "3", url], output_file)

def check_server_vulnerabilities(url, output_file):
    output_file.write("## Checking Server Vulnerabilities\n")
    run_command(["nmap", "-sV", "--script=vuln", url], output_file)

def run_automatic_scanners(url, output_file):
    output_file.write("## Running Automatic Scanners\n")
    run_command(["nikto", "-h", url], output_file)
    run_command(["wapiti", "-u", url], output_file)
    run_command(["nuclei", "-t", "nuclei-templates", "-u", url], output_file)

def initial_checks(url, output_file):
    output_file.write("## Performing Initial Checks\n")
    paths = ["/robots.txt", "/sitemap.xml", "/crossdomain.xml", "/clientaccesspolicy.xml", "/.well-known/"]
    for path in paths:
        run_command(["curl", "-I", f"{url}{path}"], output_file)

def check_ssl_tls(url, output_file):
    output_file.write("## Checking SSL/TLS\n")
    run_command(["testssl.sh", url], output_file)

def spider_website(url, output_file):
    output_file.write("## Spidering Website\n")
    run_command(["gospider", "-s", url], output_file)

def brute_force_directories(url, output_file):
    output_file.write("## Brute Forcing Directories\n")
    run_command(["feroxbuster", "-u", url], output_file)

def check_file_backups(url, output_file):
    output_file.write("## Checking File Backups\n")
    run_command(["bfac", "-u", url], output_file)

def discover_parameters(url, output_file):
    output_file.write("## Discovering Parameters\n")
    run_command(["arjun", "-u", url], output_file)

def check_web_vulnerabilities(url, output_file):
    output_file.write("## Checking Web Vulnerabilities\n")
    output_file.write("Perform manual checks for XSS, CSRF, SQLi, etc.\n")

def check_403_bypasses(url, output_file):
    output_file.write("## Checking for 403 Bypass Techniques\n")
    headers = {
        "X-Original-URL": "/admin",
        "X-Rewrite-URL": "/admin",
        "X-Forwarded-For": "127.0.0.1",
        "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.1 Mobile/15E148 Safari/604.1"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        output_file.write("Potential 403 bypass found. Check manually.\n")

def check_ntlm_info_disclosure(url, output_file):
    output_file.write("## Checking for NTLM Authentication Info Disclosure\n")
    headers = {"Authorization": "NTLM TlRMTVNTUAABAAAAB4IIAAAAAAAAAAAAAAAAAAAAAAA="}
    response = requests.get(url, headers=headers)
    if "WWW-Authenticate" in response.headers:
        output_file.write("NTLM info disclosure found. Check 'WWW-Authenticate' header.\n")

def check_js_files(url, output_file):
    output_file.write("## Analyzing JavaScript Files\n")
    run_command(["subjs", "-u", url], output_file)
    run_command(["linkfinder", "-i", url, "-d"], output_file)

def check_api_endpoints(url, output_file):
    output_file.write("## Checking for API Endpoints\n")
    run_command(["ffuf", "-w", "/path/to/api_wordlist.txt", "-u", f"{url}/FUZZ", "-mc", "200,201,204"], output_file)

def directory_fuzzing(url, output_file):
    output_file.write("## Directory Fuzzing\n")
    run_command(["ffuf", "-w", "/opt/useful/SecLists/Discovery/Web-Content/directory-list-2.3-small.txt:FUZZ", "-u", f"{url}/FUZZ"], output_file)

def extension_fuzzing(url, output_file):
    output_file.write("## Extension Fuzzing\n")
    run_command(["ffuf", "-w", "/opt/useful/SecLists/Discovery/Web-Content/web-extensions.txt:FUZZ", "-u", f"{url}/indexFUZZ"], output_file)

def page_fuzzing(url, output_file):
    output_file.write("## Page Fuzzing\n")
    run_command(["ffuf", "-w", "/opt/useful/SecLists/Discovery/Web-Content/directory-list-2.3-small.txt:FUZZ", "-u", f"{url}/FUZZ.php"], output_file)

def recursive_fuzzing(url, output_file):
    output_file.write("## Recursive Fuzzing\n")
    run_command(["ffuf", "-w", "/opt/useful/SecLists/Discovery/Web-Content/directory-list-2.3-small.txt:FUZZ", "-u", f"{url}/FUZZ", "-recursion", "-recursion-depth", "1", "-e", ".php", "-v"], output_file)

def subdomain_fuzzing(domain, output_file):
    output_file.write("## Subdomain Fuzzing\n")
    run_command(["ffuf", "-w", "/opt/useful/SecLists/Discovery/DNS/subdomains-top1million-5000.txt:FUZZ", "-u", f"https://FUZZ.{domain}/"], output_file)

def vhost_fuzzing(url, output_file):
    output_file.write("## VHost Fuzzing\n")
    parsed_url = urlparse(url)
    run_command(["ffuf", "-w", "/opt/useful/SecLists/Discovery/DNS/subdomains-top1million-5000.txt:FUZZ", "-u", url, "-H", f"Host: FUZZ.{parsed_url.netloc}", "-fs", "900"], output_file)

def parameter_fuzzing_get(url, output_file):
    output_file.write("## Parameter Fuzzing - GET\n")
    run_command(["ffuf", "-w", "/opt/useful/SecLists/Discovery/Web-Content/burp-parameter-names.txt:FUZZ", "-u", f"{url}?FUZZ=key", "-fs", "xxx"], output_file)

def parameter_fuzzing_post(url, output_file):
    output_file.write("## Parameter Fuzzing - POST\n")
    run_command(["ffuf", "-w", "/opt/useful/SecLists/Discovery/Web-Content/burp-parameter-names.txt:FUZZ", "-u", url, "-X", "POST", "-d", "FUZZ=key", "-H", "Content-Type: application/x-www-form-urlencoded", "-fs", "xxx"], output_file)

def value_fuzzing(url, output_file):
    output_file.write("## Value Fuzzing\n")
    run_command(["ffuf", "-w", "wordlist.txt:FUZZ", "-u", url, "-X", "POST", "-d", "id=FUZZ", "-H", "Content-Type: application/x-www-form-urlencoded", "-fs", "xxx"], output_file)

def main(args):
    url = fix_url_scheme(args.url)
    url = check_redirect(url)
    parsed_url = urlparse(url)

    with open(args.output, 'w') as output_file:
        if args.all or args.whatweb:
            identify_technologies(url, output_file)
        if args.all or args.nmap:
            check_server_vulnerabilities(parsed_url.netloc, output_file)
        if args.all or args.auto_scanners:
            run_automatic_scanners(url, output_file)
        if args.all or args.initial_checks:
            initial_checks(url, output_file)
        if args.all or args.ssl_tls:
            check_ssl_tls(parsed_url.netloc, output_file)
        if args.all or args.spider:
            spider_website(url, output_file)
        if args.all or args.brute_force:
            brute_force_directories(url, output_file)
        if args.all or args.file_backups:
            check_file_backups(url, output_file)
        if args.all or args.discover_params:
            discover_parameters(url, output_file)
        if args.all or args.web_vulns:
            check_web_vulnerabilities(url, output_file)
        if args.all or args.bypass_403:
            check_403_bypasses(url, output_file)
        if args.all or args.ntlm_info:
            check_ntlm_info_disclosure(url, output_file)
        if args.all or args.js_files:
            check_js_files(url, output_file)
        if args.all or args.api_endpoints:
            check_api_endpoints(url, output_file)
        if args.all or args.fuzz_dirs:
            directory_fuzzing(url, output_file)
        if args.all or args.fuzz_ext:
            extension_fuzzing(url, output_file)
        if args.all or args.fuzz_pages:
            page_fuzzing(url, output_file)
        if args.all or args.fuzz_recursive:
            recursive_fuzzing(url, output_file)
        if args.all or args.fuzz_subdomains:
            subdomain_fuzzing(parsed_url.netloc, output_file)
        if args.all or args.fuzz_vhosts:
            vhost_fuzzing(url, output_file)
        if args.all or args.fuzz_params_get:
            parameter_fuzzing_get(url, output_file)
        if args.all or args.fuzz_params_post:
            parameter_fuzzing_post(url, output_file)
        if args.all or args.fuzz_values:
            value_fuzzing(url, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Web Penetration Testing Methodology Script")
    parser.add_argument("url", help="Target URL to test")
    parser.add_argument("-o", "--output", default="output.txt", help="Output file name")
    parser.add_argument("-a", "--all", action="store_true", help="Run all tests")
    parser.add_argument("-w", "--whatweb", action="store_true", help="Run whatweb")
    parser.add_argument("-n", "--nmap", action="store_true", help="Run nmap vulnerability scan")
    parser.add_argument("-s", "--auto-scanners", action="store_true", help="Run automatic scanners")
    parser.add_argument("-i", "--initial-checks", action="store_true", help="Perform initial checks")
    parser.add_argument("-t", "--ssl-tls", action="store_true", help="Check SSL/TLS")
    parser.add_argument("-p", "--spider", action="store_true", help="Spider website")
    parser.add_argument("-b", "--brute-force", action="store_true", help="Brute force directories")
    parser.add_argument("-f", "--file-backups", action="store_true", help="Check file backups")
    parser.add_argument("-d", "--discover-params", action="store_true", help="Discover parameters")
    parser.add_argument("-v", "--web-vulns", action="store_true", help="Check web vulnerabilities")
    parser.add_argument("-y", "--bypass-403", action="store_true", help="Check for 403 bypass techniques")
    parser.add_argument("-m", "--ntlm-info", action="store_true", help="Check for NTLM info disclosure")
    parser.add_argument("-j", "--js-files", action="store_true", help="Analyze JavaScript files")
    parser.add_argument("-e", "--api-endpoints", action="store_true", help="Check for API endpoints")
    parser.add_argument("--fuzz-dirs", action="store_true", help="Fuzz directories")
    parser.add_argument("--fuzz-ext", action="store_true", help="Fuzz extensions")
    parser.add_argument("--fuzz-pages", action="store_true", help="Fuzz pages")
    parser.add_argument("--fuzz-recursive", action="store_true", help="Recursive fuzzing")
    parser.add_argument("--fuzz-subdomains", action="store_true", help="Fuzz subdomains")
    parser.add_argument("--fuzz-vhosts", action="store_true", help="Fuzz virtual hosts")
    parser.add_argument("--fuzz-params-get", action="store_true", help="Fuzz GET parameters")
    parser.add_argument("--fuzz-params-post", action="store_true", help="Fuzz POST parameters")
    parser.add_argument("--fuzz-values", action="store_true", help="Fuzz parameter values")
    
    args = parser.parse_args()
    
    main(args)
