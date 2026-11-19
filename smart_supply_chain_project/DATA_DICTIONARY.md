# Data Dictionary – supply_chain_orders.csv

| Column | Description |
|---|---|
| order_id | Unique purchase-order identifier |
| supplier_id | Supplier code |
| supplier_name | Supplier name |
| product_category | Ordered product category |
| region | Delivery region |
| shipping_mode | Standard, Express, or Economy |
| order_date | Purchase-order date |
| promised_delivery_date | Expected delivery date |
| actual_delivery_date | Recorded delivery date |
| order_quantity | Number of units ordered |
| unit_cost | Cost per unit |
| total_order_value | Quantity multiplied by unit cost |
| promised_duration_days | Promised delivery duration |
| delivery_duration_days | Actual delivery duration |
| delay_days | Days beyond promised duration; zero for on-time |
| supplier_rating | Supplier rating on a 10-point scale |
| supplier_review | Text feedback about the supplier |
| delivery_status | Target label: On-time or Delayed |

All records are synthetic and generated for this academic project.
