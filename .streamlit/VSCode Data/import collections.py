import collections


def count_companies(mushed_text, company_list):
    # Sort companies by length (longest first) to prevent partial matching
    # (e.g., counting "Apple" when the text actually says "Applebee's")
    company_list.sort(key=len, reverse=True)

    company_counts = collections.Counter()

    for company in company_list:
        count = mushed_text.count(company)
        if count > 0:
            company_counts[company] = count
            # Remove the found company from the text to avoid double-counting
            mushed_text = mushed_text.replace(company, "")

    return company_counts, mushed_text


# 1. Paste your giant block of text here
# Replace with your full text
raw_text = "National Football League - NFLCitiPNC BankTravelers..."

# 2. Add your known companies to this list.
# (I've included the major ones from your data to get you started)
companies = [
    "National Football League - NFL", "Marsh McLennan Companies (MMC)", "Delta Air Lines, Inc.",
    "JPMorgan Chase & Co.", "Starbucks Coffee Company", "Boston Consulting Group",
    "Wellington Management Company LLP", "Lincoln Financial Group", "Mondelēz International",
    "The Walt Disney Company", "Ferrara Candy Company", "Goldman Sachs & Co.",
    "Dell Technologies Inc.", "American Express Company Inc.", "Colgate-Palmolive Company",
    "Princeton University Investment Company (PRINCO)", "Makena Capital Management LLC",
    "Ernst & Young Global Limited", "Microsoft Corporation", "Boston Consulting",
    "Jane Street", "General Mills", "Autodesk", "Deloitte", "PepsiCo", "Target",
    "Amazon", "Google", "Apple", "Nike Inc.", "Comcast", "Netflix", "Visa Inc.",
    "Mars", "FICO", "UBS", "Citi", "PGIM"
    # Add the rest of your specific companies here...
]

# 3. Run the counting function
counts, leftover_text = count_companies(raw_text, companies)

# 4. Print the results in descending order
print("--- COMPANY MENTIONS ---")
for company, count in counts.most_common():
    print(f"{company}: {count}")

print("\n--- UNMATCHED LEFTOVER TEXT ---")
# This prints whatever text wasn't matched so you can see what companies
# you still need to add to your 'companies' list above!
print(leftover_text)
