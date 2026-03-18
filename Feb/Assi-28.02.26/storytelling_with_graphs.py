import os
import statistics
from collections import Counter

import matplotlib.pyplot as plt


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def create_bar_chart(months, monthly_sales, output_dir: str) -> str:
    plt.figure(figsize=(10, 5))
    bars = plt.bar(months, monthly_sales, color="#2F6CAD")
    plt.title("Monthly Sales Trend (Bar Chart)")
    plt.xlabel("Month")
    plt.ylabel("Sales (in $1,000)")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, value in zip(bars, monthly_sales):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    out_path = os.path.join(output_dir, "bar_chart.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def create_pie_chart(category_sales, output_dir: str) -> str:
    labels = list(category_sales.keys())
    values = list(category_sales.values())

    plt.figure(figsize=(7, 7))
    plt.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=["#F39C12", "#3498DB", "#2ECC71", "#9B59B6"],
    )
    plt.title("Sales Contribution by Category (Pie Chart)")
    plt.axis("equal")

    out_path = os.path.join(output_dir, "pie_chart.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def create_histogram(order_values, output_dir: str) -> str:
    plt.figure(figsize=(9, 5))
    plt.hist(order_values, bins=12, color="#16A085", edgecolor="black", alpha=0.8)
    plt.title("Distribution of Order Values (Histogram)")
    plt.xlabel("Order Value ($)")
    plt.ylabel("Number of Orders")
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    out_path = os.path.join(output_dir, "histogram.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def write_data_story(
    months, monthly_sales, category_sales, order_values, output_dir: str
) -> str:
    first_month, last_month = months[0], months[-1]
    first_sales, last_sales = monthly_sales[0], monthly_sales[-1]
    growth_pct = ((last_sales - first_sales) / first_sales) * 100

    best_month_index = monthly_sales.index(max(monthly_sales))
    worst_month_index = monthly_sales.index(min(monthly_sales))
    best_month = months[best_month_index]
    worst_month = months[worst_month_index]

    total_category_sales = sum(category_sales.values())
    top_category = max(category_sales, key=category_sales.get)
    top_category_share = (category_sales[top_category] / total_category_sales) * 100

    rounded_orders = [10 * round(v / 10) for v in order_values]
    common_bucket = Counter(rounded_orders).most_common(1)[0][0]
    mean_order = statistics.mean(order_values)
    median_order = statistics.median(order_values)

    story = (
        "Data Story: Sales and Customer Buying Patterns\n\n"
        f"From {first_month} to {last_month}, monthly sales increased from "
        f"${first_sales}k to ${last_sales}k, a growth of {growth_pct:.1f}%.\n"
        f"The strongest month was {best_month}, while {worst_month} had the lowest sales.\n\n"
        f"In category mix, {top_category} led with {top_category_share:.1f}% of total category sales, "
        "showing where the business currently has the strongest demand.\n\n"
        "The order-value histogram suggests most purchases cluster around the "
        f"${common_bucket} range, with an average order value of ${mean_order:.1f} "
        f"and median ${median_order:.1f}. This indicates a reliable mid-value customer segment, "
        "with fewer high-value purchases.\n\n"
        "Overall trend: sales momentum is positive, category concentration is clear, and "
        "customer spend is centered around medium-sized orders."
    )

    out_path = os.path.join(output_dir, "data_story.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(story)
    return out_path


def main() -> None:
    output_dir = "output_storytelling"
    ensure_output_dir(output_dir)

    # Sample dataset (can be replaced with real project data).
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    monthly_sales = [120, 128, 125, 138, 145, 160, 170, 166, 178, 190, 205, 220]

    category_sales = {
        "Electronics": 420,
        "Clothing": 300,
        "Home Decor": 220,
        "Books": 160,
    }

    order_values = [
        38,
        42,
        45,
        49,
        52,
        55,
        57,
        60,
        62,
        65,
        67,
        70,
        72,
        74,
        76,
        78,
        80,
        82,
        84,
        86,
        88,
        90,
        92,
        95,
        98,
        104,
        110,
        118,
    ]

    bar_path = create_bar_chart(months, monthly_sales, output_dir)
    pie_path = create_pie_chart(category_sales, output_dir)
    hist_path = create_histogram(order_values, output_dir)
    story_path = write_data_story(
        months, monthly_sales, category_sales, order_values, output_dir
    )

    print("Generated files:")
    print(f"- {bar_path}")
    print(f"- {pie_path}")
    print(f"- {hist_path}")
    print(f"- {story_path}")


if __name__ == "__main__":
    main()
