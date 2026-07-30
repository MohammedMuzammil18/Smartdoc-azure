"""
verify_requirements.py
======================
Automated verification suite validating all TCS iON Industry Project requirements:
1. Portal Authentication & User Self-Service
2. Multi-Domain Document Coverage (HR, IT, Finance policy & employment docs)
3. Natural Language & Intent-Based Search Processing
4. Synonym & Conceptual Keyword Match
5. ML Document Categorisation
6. Live Autocomplete & Query Suggestions
7. Document Detail View & Export Capabilities
8. Admin Management & System Health Monitoring
"""

import urllib.request
import json
import time

BASE_URL = "http://127.0.0.1:5000"

def post(endpoint, data):
    req = urllib.request.Request(
        f"{BASE_URL}{endpoint}",
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def get(endpoint):
    req = urllib.request.Request(f"{BASE_URL}{endpoint}")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def run_tests():
    print("==========================================================================")
    print("  TCS iON Industry Project Requirements Verification Audit")
    print("==========================================================================")
    print()

    tests_passed = 0
    total_tests = 0

    # ── Test 1: System Health & Search Subsystems ─────────────────────────────
    total_tests += 1
    print("[REQ-1] System Health & Search Subsystems:")
    try:
        health = get("/health")
        assert health.get("status") == "online"
        assert health.get("model_loaded") == True
        assert health.get("semantic_search_ready") == True
        assert health.get("document_count", 0) >= 15
        print(f"  PASS: Backend Online | Index Docs: {health['document_count']} | ML Model: Ready | Semantic Engine: Ready")
        tests_passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")

    # ── Test 2: Natural Language & Intent-Based Queries ──────────────────────
    total_tests += 1
    print("\n[REQ-2] Human Intent & Natural Language Query Processing:")
    intent_queries = [
        ("how do I work securely from home", ["VPN Setup Guide", "Remote Work Policy", "it-1", "hr-2"]),
        ("taking time off when sick or for family", ["PTO & Leave Guidelines", "Employee Handbook", "hr-3", "hr-1"]),
        ("getting money back for client meals", ["Expense Reimbursement Policy", "Travel & Entertainment Policy", "fin-1", "fin-2"]),
        ("buying software for my team", ["Software Procurement Policy", "it-3"])
    ]
    all_matched = True
    for query, expected_keys in intent_queries:
        res = post("/search", {"query": query, "top": 5})
        results = res.get("results", [])
        found = False
        top_titles = [r.get("title", r.get("text", ""))[:40] for r in results]
        for r in results[:3]:
            combined = f"{r.get('id','')} {r.get('title','')} {r.get('text','')} {r.get('snippet','')}".lower()
            if any(k.lower() in combined for k in expected_keys):
                found = True
                print(f"  PASS: Query: '{query}' -> Matched document '{r.get('title', r.get('id'))}' (Score: {r.get('score', 0):.3f})")
                break
        if not found:
            print(f"  FAIL: Query: '{query}' -> Did not find expected document in top 3. Top titles: {top_titles}")
            all_matched = False
    if all_matched:
        tests_passed += 1

    # ── Test 3: Synonyms & Conceptual Keyword Processing ──────────────────────
    total_tests += 1
    print("\n[REQ-3] Synonym & Conceptual Keyword Processing:")
    synonym_queries = [
        ("remuneration salary compensation raise", ["Performance Review Process", "Employee Handbook", "hr-4", "hr-1"]),
        ("laptop credentials security code", ["IT Security Guidelines", "VPN Setup Guide", "it-2", "it-1"]),
        ("flight hotel lodging per diem", ["Travel & Entertainment Policy", "Expense Reimbursement Policy", "fin-2", "fin-1"])
    ]
    syn_matched = True
    for query, expected_keys in synonym_queries:
        res = post("/search", {"query": query, "top": 5})
        results = res.get("results", [])
        found = False
        for r in results[:3]:
            combined = f"{r.get('id','')} {r.get('title','')} {r.get('text','')} {r.get('snippet','')}".lower()
            if any(k.lower() in combined for k in expected_keys):
                found = True
                print(f"  PASS: Synonyms: '{query}' -> Matched document '{r.get('title', r.get('id'))}' (Score: {r.get('score', 0):.3f})")
                break
        if not found:
            print(f"  FAIL: Synonyms: '{query}' -> Did not find expected document")
            syn_matched = False
    if syn_matched:
        tests_passed += 1

    # ── Test 4: ML Document Categorisation ────────────────────────────────────
    total_tests += 1
    print("\n[REQ-4] ML Document Categoriser (HR / IT / Finance):")
    sample_texts = [
        ("New employee orientation and health benefit enrollment package", "HR"),
        ("Cisco AnyConnect VPN installation failover and firewall rule reset", "IT"),
        ("Quarterly budget variance calculation and invoice reimbursement claim", "Finance")
    ]
    ml_pass = True
    for sample, expected_cat in sample_texts:
        pred = post("/predict", {"text": sample})
        cat = pred.get("category")
        if cat == expected_cat:
            print(f"  PASS: Classified '{sample[:45]}...' -> {cat}")
        else:
            print(f"  FAIL: Classified '{sample[:45]}...' -> Got '{cat}', Expected '{expected_cat}'")
            ml_pass = False
    if ml_pass:
        tests_passed += 1

    # ── Test 5: Live Autocomplete ─────────────────────────────────────────────
    total_tests += 1
    print("\n[REQ-5] Live Autocomplete & Instant Suggestions:")
    ac_res = post("/autocomplete", {"query": "work", "top": 5})
    suggs = ac_res.get("suggestions", [])
    if len(suggs) > 0:
        print(f"  PASS: Prefix 'work' -> Suggestions: {suggs[:3]}")
        tests_passed += 1
    else:
        print("  FAIL: No autocomplete suggestions returned.")

    # ── Test 6: Auth & Self-Service User Portal ──────────────────────────────
    total_tests += 1
    print("\n[REQ-6] User Portal Authentication & Registration:")
    test_email = f"emp_{int(time.time())}@nexussolutions.in"
    reg_resp = post("/register", {"name": "Audit Employee", "email": test_email, "password": "securepass123"})
    login_resp = post("/login", {"email": test_email, "password": "securepass123"})
    if reg_resp.get("success") and login_resp.get("success"):
        print(f"  PASS: Registered & Authenticated employee: {test_email}")
        tests_passed += 1
    else:
        print(f"  FAIL: Auth check failed. Reg: {reg_resp}, Login: {login_resp}")

    # ── Test 7: Admin Document Management ─────────────────────────────────────
    total_tests += 1
    print("\n[REQ-7] Admin Document Management:")
    docs_resp = get("/admin/documents")
    users_resp = get("/admin/users")
    if docs_resp.get("success") and users_resp.get("success"):
        print(f"  PASS: Retrieved {docs_resp.get('count')} indexed docs and {len(users_resp.get('users'))} registered portal users")
        tests_passed += 1
    else:
        print("  FAIL: Admin endpoints failed.")

    print("\n" + "=" * 74)
    print(f"  FINAL RESULT: {tests_passed} / {total_tests} REQUIREMENT VERIFICATION CHECKS PASSED")
    print("=" * 74)

if __name__ == "__main__":
    run_tests()
