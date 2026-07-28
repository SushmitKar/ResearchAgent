import sys
from dotenv import load_dotenv

load_dotenv()

from agent import run_agent


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("What would you like me to research? ")

    print(f"\nResearching: {query}\n")

    result = run_agent(query)

    if result is None:
        print("\nSorry, the agent could not produce a valid report. Check the logs above.")
        return

    print("\n=== FINAL REPORT ===\n")
    print(f"Company: {result.company}")
    print(f"Overview: {result.overview}")
    print(f"Headquarters: {result.headquarters}")
    print(f"Founded: {result.founded}")
    print(f"Products/Services: {', '.join(result.product_services)}")
    print(f"Sources: {', '.join(result.sources)}")

    with open("report.md", "w", encoding="utf-8") as f:
        f.write(f"# {result.company}\n\n")
        f.write("## Overview\n")
        f.write(f"{result.overview}\n\n")

        if result.headquarters:
            f.write(f"**Headquarters:** {result.headquarters}\n\n")
        if result.founded:
            f.write(f"**Founded:** {result.founded}\n\n")

        f.write("## Products & Services\n")
        for product in result.product_services:
            f.write(f"- {product}\n")

        f.write("\n## Sources\n")
        for source in result.sources:
            f.write(f"- {source}\n")

if __name__ == "__main__":
    main()