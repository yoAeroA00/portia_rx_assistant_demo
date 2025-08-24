import platform
import subprocess
from typing import Any, Dict, List, Union

from dotenv import load_dotenv
from portia import (
    ActionClarification,
    Config,
    Input,
    LogLevel,
    MultipleChoiceClarification,
    PlanBuilderV2,
    PlanRunState,
    Portia,
    StepOutput,
)
from portia.cli import CLIExecutionHooks
from portia.open_source_tools.browser_tool import (
    BrowserInfrastructureOption,
    BrowserTool,
)
from portia.open_source_tools.search_tool import SearchTool
from pydantic import BaseModel, Field

load_dotenv(override=True)

config = Config.from_default(default_log_level=LogLevel.INFO)
browser_tool = BrowserTool(infrastructure_option=BrowserInfrastructureOption.LOCAL)
portia = Portia(
    config=config,
    tools=[browser_tool, SearchTool()],
    execution_hooks=CLIExecutionHooks(),
)

sources_config = [
    {"key": "1mg", "name": "1mg.com", "url": "https://www.1mg.com/"},
    {"key": "netmeds", "name": "netmeds.com", "url": "https://www.netmeds.com/"},
    {"key": "pharmeasy", "name": "pharmeasy.in", "url": "https://pharmeasy.in/"},
    {"key": "truemeds", "name": "truemeds.in", "url": "https://www.truemeds.in/"},
    {"key": "apollo", "name": "apollopharmacy.in", "url": "https://www.apollopharmacy.in/"},
]


class MedicinesList(BaseModel):
    """A model to hold the list of medicines parsed from the user's query."""

    medicines: List[str] = Field(description="A list of medicine names extracted from the user's initial query.")


class WebsitesList(BaseModel):
    """A model to hold the list of websites selected by the user."""

    websites: List[str] = Field(description="A list of website domains (e.g., '1mg.com') selected by the user for searching.")


class PincodeResult(BaseModel):
    """A model to hold the location details found automatically by the search tool."""

    location_name: str | None = Field(description="The geographical name of the location (e.g., 'Mumbai') found by the search tool.")
    pincode: str | None = Field(description="The postal code (e.g., '400001') corresponding to the location, found by the search tool.")


class ProductVariant(BaseModel):
    """Represents a single, specific medicine product found on a website."""

    name: str = Field(description="The full, official name of the medicine variant as displayed on the website (e.g., 'Dolo 650 Tablet').")
    retail_price: str | None = Field(description="The original Maximum Retail Price (MRP) of the product, often shown as struck-through.")
    discounted_price: str | None = Field(description="The final price after any discounts have been applied; this is the price the customer pays.")
    quantity: str | None = Field(description="The pack size of the medicine, such as 'strip of 15 tablets' or 'box of 1 bottle'.")
    manufacturer_marketer: str | None = Field(description="The name of the company that manufactures or markets the product.")
    url: str = Field(description="The direct web address (URL) leading to the product's detail page.")
    is_out_of_stock: bool = Field(description="A boolean value indicating the product's availability. `True` if unavailable, `False` if available for purchase.")


class ConsolidatedSearchResults(BaseModel):
    """
    A schema to structure the complete, raw results from a single website's search.
    This is the expected output from each browser-based search step.
    """

    search_results: Dict[str, Dict[str, List[ProductVariant]]] = Field(
        description="A nested dictionary holding all scraped product information. The structure is: {medicine_name: {website_name: [list_of_products]}}."
    )


def cleanup_browser_processes():
    if platform.system() == "Windows":
        for browser_exe in ["chrome.exe", "msedge.exe", "chromium.exe"]:
            try:
                subprocess.run(
                    f"taskkill /F /IM {browser_exe} /T",
                    shell=True,
                    check=False,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except Exception:
                pass
    return {"status": "Cleanup function executed successfully."}


corePlanBuilder = PlanBuilderV2("Medicine Ordering AI Agent")

corePlanBuilder.input(
    name="user_medicines_query",
    description="The user's natural language query specifying the medicines they want to search for. For example: 'I need to buy dolo 650 and crocin advance'.",
)
corePlanBuilder.llm_step(
    step_name="create_medicines_list",
    task="From the user's query, identify and extract the names of all the medicines mentioned. Return them as a clean list of strings in the `medicines` field.",
    inputs=[Input("user_medicines_query")],
    output_schema=MedicinesList,
)
corePlanBuilder.input(
    name="user_websites_selection",
    description="The user's selection of which websites to search for the medicines on. This is provided as a list of website keys. For example: ['pharmeasy', 'truemeds'].",
)
corePlanBuilder.llm_step(
    step_name="create_websites_list",
    task="Based on the user's selection of websites, create a list of the corresponding website keys. The available sources are: 1mg, netmeds, pharmeasy, truemeds, apollo. For example, if the user selects 'pharmeasy and truemeds', the output should be a list `['pharmeasy', 'truemeds']`.",
    inputs=[Input("user_websites_selection")],
    output_schema=WebsitesList,
)
corePlanBuilder.input(
    name="location_query",
    description="The user's desired delivery location, which can be a city name, a full address, or just a pincode.",
)
corePlanBuilder.single_tool_agent_step(
    step_name="find_pincode",
    task="Find the pincode for the location or find location if pincode is given.",
    tool="search_tool",
    inputs=[Input("location_query")],
    output_schema=PincodeResult,
)

SEARCH_TASK_TEMPLATE = """
### ROLE ###
You are an expert-level, automated web data extraction agent.

### CONTEXT ###
You have been given a list of medicines to find. You also have the user's location details, which could include a `location_name` and a `pincode`. One or both may be available.

### INSTRUCTIONS ###
**1. Go to Website:**
- Navigate to: '{url}'.

**2. Set Delivery Location (Crucial First Step):**
- After the page loads, your first priority is to set the delivery location.
- Look for an element like "Set Location" or "Deliver to".
- **Intelligently decide what to enter:** Observe the input field. If it asks for a "Pincode", use the `pincode`. If it asks for a "City" or "Location", use the `location_name`.
- If only one piece of data is available, use that, given it is acceptable.
- If BOTH `location_name` and `pincode` are missing, you MUST skip this step.

**3. Search, Analyze, Extract, and Consolidate:**
- For each medicine, type the exact query into the search bar and press 'Enter'.
- From the search results, filter products and extract: 'name', 'quantity', 'retail_price', 'discounted_price', 'manufacturer_marketer', 'url', and 'is_out_of_stock' (boolean).
- Compile all data into a single JSON object with a top-level "search_results" key.

**4. Finalize:**
- Close the browser.
"""

step_outputs_for_summary = []
for source in sources_config:
    search_step_name = f"product_search_{source['key']}"
    (
        corePlanBuilder.if_(
            condition=lambda websites_list_output, key=source["key"]: any(key in w for w in websites_list_output.websites),
            args={"websites_list_output": StepOutput("create_websites_list")},
        )
        .single_tool_agent_step(
            step_name=search_step_name,
            task=SEARCH_TASK_TEMPLATE.format(url=source["url"], website_name=source["name"]),
            tool="browser_tool",
            inputs=[
                StepOutput("create_medicines_list"),
            ],
            output_schema=ConsolidatedSearchResults,
        )
        .function_step(step_name=f"cleanup_{source['key']}", function=cleanup_browser_processes)
        .endif()
    )
    step_outputs_for_summary.append(StepOutput(search_step_name))

corePlanBuilder.llm_step(
    step_name="summarize_best_choices",
    task="""
        ### ROLE ###
        You are a helpful assistant that analyzes medicine search results to provide the best purchasing plan.

        ### CONTEXT ###
        You have the search results for a list of medicines from one or more online pharmacies. Your goal is to help the user make the most cost-effective and convenient choice.

        ### INSTRUCTIONS ###
        1.  **Analyze and Consolidate:** Go through the search results from all websites. For each requested medicine, identify the best option (lowest price, in-stock).
        2.  **Find the Optimal Single-Store Plan:** Determine if there is one single website that has all the requested medicines in stock. If so, calculate the total cost from that single store. Identify the store that offers the lowest total price for the complete order.
        3.  **Create a Multi-Store Plan:** If no single store has all the medicines, devise a purchasing plan that sources all the medicines from the minimum number of stores, prioritizing the lowest cost for each item.
        4.  **Structure Your Recommendation:** Present your findings in a clear, easy-to-read format using Markdown. Use the following structure:

            *   **"Best Recommendation":**
                *   If all medicines are available on a single website, state which one is the cheapest and the total cost.
                *   If not, recommend the best multi-store purchasing plan. For example: "Order Dolo 650 from PharmEasy and Crocin Advance from TrueMeds for a total cost of..."

            *   **"Sequential Purchasing Plan":**
                *   Provide a step-by-step guide for the user to follow.

            *   **"Price Comparison":**
                *   Provide a detailed breakdown for each medicine in a Markdown table, showing the prices across all websites where it was found, to justify your recommendation. Clearly mark the best price for each.

            *   **"Out of Stock Items":**
                *   List any medicines that were not available on any of the selected websites.
        """,
    inputs=step_outputs_for_summary,
)

corePlan = corePlanBuilder.final_output(summarize=False).build()


def handle_clarifications(portia, plan_run):
    """
    Universal clarification handler: Dynamically processes all outstanding clarifications.
    """
    while plan_run.state == PlanRunState.NEED_CLARIFICATION:
        clarifications = plan_run.get_outstanding_clarifications()

        for clarification in clarifications:
            print("\n--- Clarification Requested ---")
            print(f"Step: {clarification.step}")
            print(f"Guidance: {clarification.user_guidance}\n")

            if isinstance(clarification, MultipleChoiceClarification):
                print("Please choose one of the following options:")
                for idx, option in enumerate(clarification.options, start=1):
                    print(f"  {idx}. {option}")

                while True:
                    choice = input("Enter the option number: ").strip()
                    try:
                        choice_idx = int(choice) - 1
                        if 0 <= choice_idx < len(clarification.options):
                            user_input = clarification.options[choice_idx]
                            break
                        else:
                            print("Invalid choice, please try again.")
                    except ValueError:
                        print("Please enter a valid number.")

                plan_run = portia.resolve_clarification(clarification, user_input, plan_run)

            elif isinstance(clarification, ActionClarification):
                print(f"Action required! Please complete the task at:\n{clarification.action_url}")
                input("Press ENTER once the action is completed...")
                plan_run = portia.wait_for_ready(plan_run)

            else:
                user_input = input("Your response: ").strip()
                plan_run = portia.resolve_clarification(clarification, user_input, plan_run)

        plan_run = portia.resume(plan_run)

    return plan_run


if __name__ == "__main__":
    print("--- Medicine Ordering AI Agent ---")

    # Get inputs from the user
    user_medicines_query = input("Enter the medicines you want to search for (e.g., dolo 650, crocin advance): ")
    user_websites_selection = input("Enter the websites to search on (e.g., pharmeasy, truemeds): ")
    location_query = input("Enter your delivery location (e.g., Mumbai or 400001): ")

    # Construct the inputs dictionary
    plan_run_inputs = {
        "user_medicines_query": user_medicines_query,
        "user_websites_selection": user_websites_selection,
        "location_query": location_query,
    }

    # Run the plan with the user-provided inputs
    plan_run = portia.run_plan(
        corePlan,
        plan_run_inputs=plan_run_inputs,
    )

    plan_run = handle_clarifications(portia, plan_run)
