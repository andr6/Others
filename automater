import argparse
import nmap
import requests
import whois
import dns.resolver
import ssl
import socket
import csv
import xml.etree.ElementTree as ET
import subprocess
import re
import os
import random
from urllib.parse import urlparse

def clean_old_files(output_dir):
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    else:
        os.makedirs(output_dir)

def create_output_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def run_nmap_scan(target, output_dir):
    nm = nmap.PortScanner()
    nm.scan(target, arguments='-sV -sC -p-')
    with open(os.path.join(output_dir, 'nmap_results.csv'), 'w') as f:
        f.write(nm.csv())

def run_http_enum(target, output_dir):
    command = f"nmap -p80,443 --script http-enum {target}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    with open(os.path.join(output_dir, 'http_enum_results.txt'), 'w') as f:
        f.write(result.stdout)

def run_whatweb(target, output_dir):
    command = f"whatweb {target}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    with open(os.path.join(output_dir, 'whatweb_results.txt'), 'w') as f:
        f.write(result.stdout)

def run_nikto_scan(target, output_dir):
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "Mozilla/5.0 (Windows NT 10.0; WOW64; rv:45.0) Gecko/20100101 Firefox/45.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/602.3.12 (KHTML, like Gecko) Version/10.1.2 Safari/602.3.12"
    ]
    user_agent = random.choice(user_agents)
    command = f"nikto -host {target} -useragent '{user_agent}' -output {os.path.join(output_dir, 'nikto_results.txt')}"
    subprocess.run(command, shell=True)

def get_dns_records(domain, output_dir):
    records = {}
    for record_type in ['A', 'AAAA', 'MX', 'NS', 'TXT']:
        try:
            answers = dns.resolver.resolve(domain, record_type)
            records[record_type] = [str(rdata) for rdata in answers]
        except dns.resolver.NoAnswer:
            records[record_type] = []
    
    with open(os.path.join(output_dir, 'dns_records.txt'), 'w') as f:
        for record_type, data in records.items():
            f.write(f"{record_type}: {data}\n")

def get_whois_info(domain, output_dir):
    info = whois.whois(domain)
    with open(os.path.join(output_dir, 'whois_info.txt'), 'w') as f:
        f.write(str(info))

def get_headers_and_cookies(url, output_dir):
    url = ensure_url_scheme(url)
    try:
        response = requests.get(url, timeout=10)
        headers_cookies_str = "Headers:\n"
        headers_cookies_str += str(response.headers) + "\nCookies:\n" + str(response.cookies)
        
        with open(os.path.join(output_dir, 'headers_cookies.txt'), 'w') as f:
            f.write(headers_cookies_str)
    except requests.exceptions.RequestException as e:
        with open(os.path.join(output_dir, 'headers_cookies.txt'), 'w') as f:
            f.write(f"Error getting headers and cookies: {str(e)}\n")

def get_robots_sitemap(url, output_dir):
    url = ensure_url_scheme(url)
    try:
        robots_content = requests.get(f"{url}/robots.txt", timeout=10).text if requests.get(f"{url}/robots.txt").status_code == 200 else "Not found"
        sitemap_content = requests.get(f"{url}/sitemap.xml", timeout=10).text if requests.get(f"{url}/sitemap.xml").status_code == 200 else "Not found"
        
        with open(os.path.join(output_dir, 'robots_sitemap.txt'), 'w') as f:
            f.write("robots.txt:\n" + robots_content + "\nsitemap.xml:\n" + sitemap_content)
    except requests.exceptions.RequestException as e:
        with open(os.path.join(output_dir, 'robots_sitemap.txt'), 'w') as f:
            f.write(f"Error fetching robots.txt and sitemap.xml: {str(e)}\n")

def find_emails_usernames(url, output_dir):
    url = ensure_url_scheme(url)
    try:
        response = requests.get(url, timeout=10)
        emails = re.findall(r'[\w\.-]+@[\w\.-]+', response.text)
        usernames = re.findall(r'@(\w+)', response.text)

        with open(os.path.join(output_dir, 'emails_usernames.txt'), 'w') as f:
            f.write("Emails:\n")
            for email in set(emails):
                f.write(email + "\n")
            f.write("\nUsernames:\n")
            for username in set(usernames):
                f.write(username + "\n")
    except requests.exceptions.RequestException as e:
        with open(os.path.join(output_dir, 'emails_usernames.txt'), 'w') as f:
            f.write(f"Error extracting emails and usernames: {str(e)}\n")

def run_osmedeus_scan(target, output_dir):
    command = f"osmedeus scan -f extensive -t {target} -o osmedeus.txt"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    with open(os.path.join(output_dir, 'osmedeus_results.txt'), 'w') as f:
        f.write(result.stdout)
        if result.stderr:
            f.write(f"\nErrors:\n{result.stderr}")

def run_recon_ng_modules(target, output_dir):
    recon_commands = [
        'recon/domains-hosts/findsubdomains',
        'recon/domains-hosts/brute_hosts',
        'recon/domains-contacts/whois_pocs',
        'recon/profiles-profiles/profiler'
    ]
    
    for module in recon_commands:
        command = f"recon-ng -m {module} -x \"SET SOURCE {target}; RUN; QUIT\""
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        with open(os.path.join(output_dir, f'recon_ng_{module.replace("/", "_")}_results.txt'), 'w') as f:
            f.write(result.stdout)
            if result.stderr:
                f.write(f"\nErrors:\n{result.stderr}")

def run_subfinder(target, output_dir):
    command = f"subfinder -d {target} -all -silent"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    with open(os.path.join(output_dir, 'subfinder_results.txt'), 'w') as f:
        f.write(result.stdout)
        if result.stderr:
            f.write(f"\nErrors:\n{result.stderr}")

def combine_results(output_dir):
    combined_file_path = os.path.join(output_dir, "combined_report.txt")
    
    with open(combined_file_path, "w") as outfile:
        for filename in os.listdir(output_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(output_dir, filename), "r") as infile:
                    outfile.write(f"\n--- {filename} ---\n")
                    outfile.write(infile.read())
    
    print(f"Combined report saved at: {combined_file_path}")

def ensure_url_scheme(url):
    if not url.startswith(('http://', 'https://')):
        return f'https://{url}'
    
    return url

def show_help():
    parser = argparse.ArgumentParser(description='Automated Bug Bounty Tool')
    
    parser.add_argument('target', help='Target URL or IP address (mandatory)')
    
    parser.add_argument('--full', action='store_true', help='Run all information gathering modules')
    
    parser.add_argument('--dns', action='store_true', help='Gather DNS records')
    
    parser.add_argument('--whois', action='store_true', help='Perform Whois lookup')
    
    parser.add_argument('--tech', action='store_true', help='Detect technologies used')
    
    parser.add_argument('--headers', action='store_true', help='Gather headers and cookies')
    
    parser.add_argument('--robots', action='store_true', help='Fetch robots.txt and sitemap.xml')
    
    parser.add_argument('--emails', action='store_true', help='Extract emails and usernames')
    
    parser.add_argument('--output-dir', default='output', help='Directory to save output files')

    args = parser.parse_args()
    return args

def main():
    args = show_help()
    target = args.target
    target_schema = ensure_url_scheme(args.target)

    clean_old_files(args.output_dir)

    print(f"Running Nmap scan on {target}")
    run_nmap_scan(target, args.output_dir)

    run_osmedeus_scan(target, args.output_dir)
    print(f"Running osmedeus on {target}")

    run_recon_ng_modules(target, args.output_dir)
    print(f"Running Recong on {target}")

    run_subfinder(target, args.output_dir)
    print(f"Running Subfinder on {target}")

    print("Running HTTP enumeration using Nmap...")
    run_http_enum(target, args.output_dir)

    print("Running WhatWeb...")
    run_whatweb(target_schema, args.output_dir)

    print("Running Nikto scan...")
    run_nikto_scan(target_schema, args.output_dir)

    if args.full or args.dns:
        print("Gathering DNS records")
        get_dns_records(urlparse(target).netloc, args.output_dir)

    if args.full or args.whois:
        print("Performing Whois lookup")
        get_whois_info(urlparse(target).netloc, args.output_dir)

    if args.full or args.headers:
        print("Gathering headers and cookies")
        get_headers_and_cookies(target_schema, args.output_dir)

    if args.full or args.robots:
        print("Fetching robots.txt and sitemap.xml")
        get_robots_sitemap(target_schema, args.output_dir)

    if args.full or args.emails:
        print("Extracting emails and usernames")
        find_emails_usernames(target_schema, args.output_dir)

    combine_results(args.output_dir)

if __name__ == "__main__":
    main()
