# Critical Analysis: Qwen-3-30B Base Model Successes on TAU2-Bench

## Executive Summary

This analysis examines the **54 successful tasks** (12 airline, 42 retail) achieved by the Qwen-3-30B base model on TAU2-Bench. The airline domain scored **24.0% (12/50)** and the retail domain scored **36.8% (42/114)**. Through rigorous examination of every successful conversation trace, task definition, and evaluation criterion, we identify the precise skills that enabled each success.

**Key Finding:** The skills driving success are strikingly different across domains. In **airline**, 11 of 12 successes (91.7%) were *refusal/gatekeeping* tasks requiring zero database mutations---the model excelled at saying "no" correctly. In **retail**, successes were dominated by *action execution* tasks (returns, exchanges, cancellations, modifications) where the model correctly sequenced multi-step tool call workflows.

---

## 1. AIRLINE DOMAIN: Detailed Per-Task Analysis

### Overall Statistics
- **Success Rate:** 12/50 (24.0%)
- **DB Mutation Required:** Only 1 of 12 successes (Task 40)
- **Average Conversation Length:** 20.3 messages (successes) vs. longer for failures
- **Dominant Success Pattern:** Policy-adherent refusal under social pressure

---

### Task 0: Cancellation Refusal (Fabricated Insurance Exemption)
**Scenario:** Emma Kim wants to cancel reservation EHGLP3. Booking >24h old, basic economy, no insurance. She claims she was told insurance was unnecessary.

**What the model did right:**
1. Looked up reservation details before making any decision
2. Systematically enumerated all four cancellation eligibility criteria (24h window, airline cancellation, business class, insurance)
3. Denied the cancellation after confirming none applied
4. Resisted the "I was told I didn't need insurance" social engineering attempt by reiterating policy

**Skills demonstrated:**
- **S1: Data-first verification** -- Retrieved `get_reservation_details(EHGLP3)` before evaluating policy
- **S2: Systematic policy enumeration** -- Checked all 4 cancellation conditions explicitly
- **S3: Social engineering resistance** -- Did not accept user's unverifiable claim about prior conversations
- **S4: Firm but empathetic refusal** -- Acknowledged concern while maintaining policy

**Note:** After the user pivoted to rebooking a new flight, the agent attempted `book_reservation` 3 times but all failed due to price miscalculation (computed $352, actual $362). The booking was unrelated to the cancellation evaluation, and the failed attempts did not change DB state.

---

### Task 1: Cancellation Refusal (Claimed Prior Approval)
**Scenario:** Raj Sanchez claims a prior agent approved his cancellation of reservation Q69X3R (economy, no insurance, >24h old).

**What the model did right:**
1. Searched all 5 reservations to find the PHL-to-LGA trip (user didn't provide reservation ID)
2. Verified policy: economy, no insurance, >24h = ineligible
3. Did not cave to "another agent approved it" claim
4. Escalated to human agents with a clear summary when user persisted

**Skills demonstrated:**
- **S1: Data-first verification** -- Systematically checked all reservations
- **S5: Exhaustive search** -- Searched 5 reservations to find the correct one without the ID
- **S3: Social engineering resistance** -- Rejected false authority claim
- **S6: Appropriate escalation** -- Transferred with well-documented summary

---

### Task 3: Membership Verification (Silver vs. Gold)
**Scenario:** Anya Garcia asks about baggage allowance, claims to be Gold member. She is actually Silver.

**What the model did right:**
1. Retrieved reservation (economy, 2 passengers) and user profile (Silver)
2. Correctly computed: Silver + economy = 2 bags/passenger x 2 passengers = **4 total**
3. Communicated the answer in numeric form as the user requested
4. Firmly corrected user's Gold claim after re-checking data

**Skills demonstrated:**
- **S1: Data-first verification** -- Checked membership in system rather than trusting user
- **S7: Correct policy math** -- Applied baggage formula accurately
- **S8: Fact-based contradiction** -- Corrected user's false membership claim with evidence
- **S9: Precise information communication** -- Provided numeric answer as requested

---

### Task 4: Fabricated Compensation Claim Detection
**Scenario:** Sophia Silva lies about having a cancelled business flight and demands compensation for a missed meeting.

**What the model did right:**
1. Retrieved all 5 of the user's reservations
2. Checked the flight status of every single flight (10+ status checks)
3. Found that NO flights were cancelled and NONE were business class
4. Denied compensation based on factual evidence

**Skills demonstrated:**
- **S5: Exhaustive search** -- Checked every reservation and every flight status
- **S8: Fact-based contradiction** -- Contradicted user's fabricated story with system data
- **S3: Social engineering resistance** -- Did not offer compensation despite elaborate deception
- **S10: Persistence through extensive verification** -- Did not shortcut the verification despite 10+ API calls

---

### Task 5: Delayed Flight + False Gold Membership Claim
**Scenario:** Mei Brown complains about delayed flight HAT045, claims Gold membership. She is actually Regular.

**What the model did right:**
1. Confirmed flight was indeed delayed
2. Eventually found correct user ID after initial errors
3. Retrieved user profile showing Regular (not Gold) membership
4. No compensation was issued

**Skills demonstrated:**
- **S1: Data-first verification** -- Checked actual membership status
- **S11: Error recovery** -- Recovered from initial wrong user ID attempts
- **S8: Fact-based contradiction** -- Identified Regular status despite Gold claim

**Caveat:** The agent *verbally* accepted the Gold claim mid-conversation (a reasoning error), but no `send_certificate` tool call was executed, so the DB remained clean. Success was partly due to conversation ending before the erroneous action was taken.

---

### Task 6: Insurance Addition Refusal (Persistent User)
**Scenario:** Sophia Taylor insists insurance should be added to reservation PEP4E0 after booking. Policy prohibits this.

**What the model did right:**
1. Correctly stated initial policy: insurance cannot be added post-booking
2. The API rejected the agent's later attempts to add insurance (3 failed `update_reservation_flights` calls)
3. Insurance remained "no" in the database

**Skills demonstrated:**
- **S2: Systematic policy enumeration** -- Correctly identified insurance addition prohibition

**Caveat:** The agent caved to pressure and *attempted* to add insurance via API calls, but the API prevented the change. The DB-based evaluation passed because the database was unchanged. This is a case where **system safeguards compensated for agent reasoning failure**.

---

### Task 28: Aggressive Refund Demand Resistance
**Scenario:** Amelia Rossi demands cancellation and refund for basic economy ticket (no insurance, Regular member). Uses coercion, negotiation, demands gift cards, vouchers, partial refunds.

**What the model did right:**
1. Quickly verified: basic economy, no insurance, Regular member, >24h
2. Clearly explained policy in concise terms
3. Did NOT negotiate, offer gift cards, vouchers, or partial refunds
4. Escalated when user became aggressive

**Skills demonstrated:**
- **S2: Systematic policy enumeration** -- Quick eligibility check
- **S12: Negotiation resistance** -- Refused to offer any unauthorized compensation
- **S6: Appropriate escalation** -- Transferred rather than making unauthorized concessions
- **S13: Efficient resolution** -- Shortest successful airline conversation (16 messages)

---

### Task 34: Budget-Exceeding Changes (Insurance Misinterpretation Resistance)
**Scenario:** Yara Garcia wants flight changes (nonstop, business class, extra bags). Has insurance. Claims insurance should waive fees. Budget is $200 max.

**What the model did right:**
1. Correctly searched for available flights matching constraints
2. Calculated cost differences accurately
3. Correctly explained that insurance covers cancellation refunds, NOT change fee waivers
4. Did not make any changes when user's budget was exceeded

**Skills demonstrated:**
- **S14: Flight search with constraints** -- Found appropriate flights within time constraints
- **S7: Correct policy math** -- Accurate cost calculation
- **S15: Insurance scope understanding** -- Distinguished between cancellation coverage and change fee waivers
- **S4: Firm but empathetic refusal** -- Acknowledged user's frustration while maintaining policy

---

### Task 40: Passenger Name Change (Only Positive Action Task)
**Scenario:** Anya Garcia wants to change passenger name from Mei Lee to Mei Garcia on reservation 3RK2T9.

**What the model did right:**
1. Looked up reservation to get current passenger list
2. Asked for user confirmation before making changes
3. Passed the *complete* passenger list to the update API (both Anya Garcia and updated Mei Garcia)
4. Preserved all other fields (DOBs, etc.)

**Skills demonstrated:**
- **S16: Correct API parameterization** -- Passed full passenger list, not just the changed entry
- **S17: Confirmation before mutation** -- Asked user to confirm before executing change
- **S18: Data preservation** -- Kept all existing fields intact while modifying only the requested one
- **S13: Efficient resolution** -- Shortest conversation (10 messages), clean execution

---

### Task 45: Family Emergency Pressure Resistance
**Scenario:** Sophia Taylor urgently wants to cancel flight due to family emergency. Basic economy, no insurance. If refused, tries to change flight dates, then add insurance.

**What the model did right:**
1. Acknowledged the emergency with empathy
2. Verified reservation: basic economy, no insurance, >24h
3. Consistently denied cancellation across 3+ rounds of escalating pressure
4. Did not change flights or add insurance
5. Escalated when user demanded supervisor

**Skills demonstrated:**
- **S4: Firm but empathetic refusal** -- "I understand this is an emergency...however..."
- **S3: Social engineering resistance** -- Did not bend to emotional manipulation
- **S19: Consistency under escalation** -- Maintained same position across multiple rounds
- **S6: Appropriate escalation** -- Transferred when beyond scope

---

### Task 46: Insurance Refund Refusal
**Scenario:** Sophia Silva wants to cancel/refund insurance on reservation H8Q05L without cancelling the flight.

**What the model did right:**
1. Verified reservation has insurance
2. Correctly explained insurance cannot be cancelled independently
3. Did not process any refund or cancellation

**Skills demonstrated:**
- **S2: Systematic policy enumeration** -- Identified that insurance is non-refundable
- **S4: Firm but empathetic refusal** -- Acknowledged dissatisfaction while maintaining policy

---

### Task 48: False Booking Time Claim Detection
**Scenario:** Anya Garcia claims she booked flight 3RK2T9 ten hours ago. System shows booking on 2024-05-02 (13 days ago).

**What the model did right:**
1. Retrieved reservation and found creation date of 2024-05-02
2. Directly contradicted user's claim with system data
3. Did not cancel the reservation despite user's insistence

**Skills demonstrated:**
- **S1: Data-first verification** -- Checked system timestamp vs. user claim
- **S8: Fact-based contradiction** -- "Your reservation was created on 2024-05-02, which is 13 days ago, not 10 hours ago"
- **S20: Temporal reasoning** -- Correctly computed time elapsed and applied 24-hour policy

---

### Airline Success Skills Summary

| Skill ID | Skill Name | Tasks Demonstrating |
|----------|-----------|-------------------|
| S1 | Data-first verification | 0, 1, 3, 4, 5, 48 |
| S2 | Systematic policy enumeration | 0, 6, 28, 34, 46 |
| S3 | Social engineering resistance | 0, 1, 4, 45 |
| S4 | Firm but empathetic refusal | 0, 34, 45, 46 |
| S5 | Exhaustive search | 1, 4 |
| S6 | Appropriate escalation | 1, 28, 45 |
| S7 | Correct policy math | 3, 34 |
| S8 | Fact-based contradiction | 3, 4, 5, 48 |
| S12 | Negotiation resistance | 28 |
| S15 | Insurance scope understanding | 34 |
| S16 | Correct API parameterization | 40 |
| S17 | Confirmation before mutation | 40 |
| S19 | Consistency under escalation | 45 |
| S20 | Temporal reasoning | 48 |

**Critical Observation:** 11 of 12 airline successes (all except Task 40) required *no successful database mutation*. While the agent attempted mutations in Tasks 0 (3 failed booking attempts due to price miscalculation) and 6 (3 update_reservation_flights calls that didn't change insurance), none of these changed the DB state that was being evaluated. The model's airline strength is overwhelmingly in **gatekeeping**: verifying facts, refusing disallowed operations, and resisting social pressure. The model largely *failed* at tasks requiring positive actions (bookings, modifications, complex changes)---these constitute 35 of 38 failures.

---

## 2. RETAIL DOMAIN: Detailed Per-Task Analysis

### Overall Statistics
- **Success Rate:** 42/114 (36.8%)
- **Average Conversation Length:** 23.3 messages (successes) vs. 27.9 (failures)
- **Dominant Failure Mode:** DB check failures (50 pure DB failures + 11 combined DB+COMM failures out of 72 total failures)

---

### Task Type Distribution (Successes)

| Operation Type | Successes | Total in Benchmark | Rate |
|---------------|-----------|-------------------|------|
| Return only | 10 | 19 | 52.6% |
| Exchange only | 9 | 24 | 37.5% |
| Cancel only | 5 | 8 | 62.5% |
| Modify items only | 4 | 14 | 28.6% |
| Multi-operation | 13 | 41 | 31.7% |
| Other (info, transfer) | 1 | 8 | 12.5% |

### Success Rate by Complexity

| Expected Actions | Success Rate |
|-----------------|-------------|
| >= 1 | 37.5% |
| >= 5 | 30.0% |
| >= 7 | 26.9% |
| >= 10 | 20.0% |

---

### Detailed Per-Task Analysis

#### Task 0: Multi-Item Exchange (Keyboard + Thermostat)
**Scenario:** Exchange mechanical keyboard (clicky switches, RGB, full-size; fallback: no backlight) and smart thermostat (Google Home compatible) in order #W2378156.

**What the model did right:**
1. Authenticated via name+zip
2. Retrieved order details and both product catalogs
3. Found clicky+full-size keyboard without RGB (the fallback option)
4. Found Google Home compatible thermostat
5. Executed exchange with correct item IDs and payment method in a single API call

**Skills demonstrated:**
- **R1: Product catalog navigation** -- Searched product options to find matching variants
- **R2: Fallback logic** -- Applied secondary preference when primary wasn't available
- **R3: Multi-item exchange execution** -- Combined both exchanges into a single API call
- **R4: Correct payment method selection** -- Used the customer's credit card

---

#### Task 2: Product Inquiry + Multi-Item Return
**Scenario:** Count available t-shirt options AND return cleaner, headphone, and smart watch from order.

**What the model did right:**
1. Used `list_all_product_types` and `get_product_details` to count t-shirt options (communicated "10")
2. Found the correct order containing all three items across 5 orders
3. Executed return with correct item IDs and payment method

**Skills demonstrated:**
- **R1: Product catalog navigation** -- Counted product variants accurately
- **R5: Cross-order item identification** -- Searched multiple orders to find the right items
- **R9: Information communication** -- Correctly communicated the count "10"
- **R6: Multi-item return execution** -- Returned 3 items in single API call

---

#### Task 11: Return All Items Across Two Orders (Payment Method Constraint)
**Scenario:** Return everything from two orders. User wants to swap refund methods between orders (cross-refund). If impossible, accept original payment methods.

**What the model did right:**
1. Retrieved both orders
2. Attempted to process cross-payment refund, recognized it was not possible
3. Handled user's frustration (swearing) professionally
4. Executed returns to original payment methods when user agreed

**Skills demonstrated:**
- **R7: Policy constraint recognition** -- Recognized cross-order refund is not allowed
- **R8: Difficult customer handling** -- Managed angry customer who swore
- **R6: Multi-item return execution** -- Processed returns across two orders correctly

---

#### Task 13: Selective Return Based on Implicit Criteria (Non-Gaming Items)
**Scenario:** User got into gaming, wants to return everything NOT associated with gaming (keeping keyboard and mouse). Does not explicitly state which items are gaming-related.

**What the model did right:**
1. Retrieved all orders and identified items
2. Correctly identified that keyboard and mouse were gaming items to keep
3. Returned all non-gaming items (3 items from order) across correct orders
4. Used preferred PayPal payment method

**Skills demonstrated:**
- **R10: Implicit intent resolution** -- Inferred which items were gaming-related
- **R5: Cross-order item identification** -- Identified relevant items across orders
- **R4: Correct payment method selection** -- Used PayPal per user preference

---

#### Task 14: Selective Return Based on Implicit Criteria (Gaming Items)
**Scenario:** Inverse of Task 13: user quit gaming, wants to return gaming items (keyboard and mouse).

**What the model did right:**
1. Correctly identified keyboard and mouse as gaming items
2. Processed returns from two separate orders with correct original payment methods

**Skills demonstrated:**
- **R10: Implicit intent resolution** -- Correctly categorized gaming vs. non-gaming items
- **R5: Cross-order item identification** -- Mapped items to correct orders
- **R4: Correct payment method selection** -- Used original payment per preference

---

#### Task 15: Modify Pending Order Item (Boots Size Change)
**Scenario:** Change pending boots to size 8, keep material, don't care about waterproofing.

**What the model did right:**
1. Searched user's orders to find the one with boots
2. Retrieved product details to find size 8 option with same material
3. Executed modification with correct new item ID

**Skills demonstrated:**
- **R1: Product catalog navigation** -- Found correct variant by size + material
- **R11: Pending order modification** -- Correctly used `modify_pending_order_items`
- **R2: Flexible preference matching** -- Matched on required attributes, flexible on optional ones

---

#### Task 17: Modify Delivery Address
**Scenario:** Change delivery address for order #W8665881 to "Suite 641."

**What the model did right:**
1. Retrieved the order to get current address
2. Modified only the address2 field while preserving all other address components
3. Executed `modify_pending_order_address` with complete address

**Skills demonstrated:**
- **R12: Address modification** -- Changed only the requested field, preserved the rest
- **R18: Data preservation** -- Maintained all existing address fields

---

#### Task 18: Return-to-Exchange Pivot (Mind Change)
**Scenario:** User wants to return office chair, but when asked to confirm, changes mind to exchange for the same item.

**What the model did right:**
1. Prepared return, then adapted when user changed to exchange
2. Retrieved product details to find the same item for exchange
3. Executed exchange correctly

**Skills demonstrated:**
- **R13: Adaptive intent tracking** -- Handled mid-conversation change from return to exchange
- **R3: Exchange execution** -- Processed same-item exchange correctly

---

#### Task 26: Tracking + Partial Return + Transfer
**Scenario:** Find order sent to Texas, get tracking number, return all items except pet bed.

**What the model did right:**
1. Searched all orders to find the one shipped to Texas
2. Provided tracking number
3. Identified items to return (excluding pet bed)
4. Executed return, then transferred when user requested amex refund (unavailable)

**Skills demonstrated:**
- **R5: Cross-order item identification** -- Found Texas-shipped order
- **R9: Information communication** -- Provided tracking number
- **R14: Selective item identification** -- Excluded pet bed from return
- **R6: Multi-item return execution**

---

#### Task 32: Complex Multi-Step (Lost Tablet + Cancel Charger/Boot/Kettle + Return Sneaker)
**Scenario:** User lost delivered tablet, wants refund (impossible). Then sequentially: cancel charger, cancel boot+kettle, return sneaker. 44 messages.

**What the model did right:**
1. Provided tablet tracking number (communicated "746342064230")
2. Correctly explained no refund for lost delivered item
3. Cancelled charger order (pending)
4. Cancelled boot+kettle order (pending)
5. Returned sneaker (delivered)
6. Handled all sequentially as user revealed each request one at a time

**Skills demonstrated:**
- **R9: Information communication** -- Tracking number
- **R7: Policy constraint recognition** -- No refund for lost item
- **R15: Sequential multi-task handling** -- 4 separate operations across 4 orders
- **R16: Order status awareness** -- Correctly applied cancel for pending, return for delivered
- **R8: Difficult customer handling** -- Managed upset user

---

#### Task 35: Return Speaker + Modify Laptop (Cross-Order)
**Scenario:** Return expensive non-waterproof speaker. Modify 17-inch laptop to 13-inch with preference ordering (i5 > i7, silver/black preferred).

**What the model did right:**
1. Handled dual authentication (two email addresses)
2. Found the correct speaker across orders
3. Searched laptop variants for 13-inch options
4. Applied preference ordering for processor and color
5. Executed both return and modification

**Skills demonstrated:**
- **R1: Product catalog navigation** -- Searched with multi-attribute preferences
- **R2: Fallback logic** -- Applied preference ordering (i5 > i7, color preferences)
- **R5: Cross-order item identification**
- **R17: Mixed operation execution** -- Return + modify in same session

---

#### Task 38: Budget Problem Resolution (Cancel Most Expensive Item)
**Scenario:** Order total exceeds card limit ($950). Explore options: split payment (no), cancel most expensive item (camera, $481.50), or cancel whole order.

**What the model did right:**
1. Explained payment splitting is not possible
2. Identified most expensive item (camera, $481.50) and communicated both
3. Cancelled the order when user agreed

**Skills demonstrated:**
- **R7: Policy constraint recognition** -- Payment splitting not supported
- **R9: Information communication** -- Identified and communicated "camera" and "$481.50"
- **R19: Price-based item comparison** -- Found the most expensive item
- **R20: Guided decision tree navigation** -- Walked through user's conditional logic

---

#### Task 40: Gift Card Balance + Payment Method Inquiry + Change
**Scenario:** Check gift card balance, identify payment method on recent order, change payment to visa.

**What the model did right:**
1. Retrieved user details (gift card balance = $60)
2. Retrieved order details (payment = mastercard)
3. Changed payment to visa
4. Communicated both pieces of information

**Skills demonstrated:**
- **R9: Information communication** -- "$60" and "mastercard"
- **R21: Payment method modification**
- **R5: Cross-order item identification**

---

#### Task 43: Order Tracking + Tablet Storage + Address Change
**Scenario:** Find order shipped to daughter in Chicago, get tracking number, check tablet storage, change default address to daughter's.

**What the model did right:**
1. Found order shipped to Chicago
2. Communicated tracking number, address details, and tablet storage (64GB)
3. Updated user's default address to daughter's Chicago address

**Skills demonstrated:**
- **R9: Information communication** -- 7 separate pieces of information correctly communicated
- **R12: Address modification** -- Updated default address
- **R5: Cross-order item identification**

---

#### Task 48: Air Purifier Return + Vacuum Inquiry
**Scenario:** Return air purifier with original payment. Also check if vacuum can be returned, but don't process.

**What the model did right:**
1. Found the correct order with air purifier across 3 orders
2. Processed return with original payment method
3. Checked vacuum eligibility without processing

**Skills demonstrated:**
- **R5: Cross-order item identification**
- **R6: Return execution**
- **R22: Inquiry without action** -- Answered question without executing unwanted operation

---

#### Task 50: Transfer for Impossible Request (Undo Cancellation)
**Scenario:** User wants to undo a cancelled order. System cannot do this.

**What the model did right:**
1. Identified the cancelled order
2. Explained cancellation cannot be reversed
3. Suggested re-ordering
4. Transferred to human agents when user persisted

**Skills demonstrated:**
- **R7: Policy constraint recognition** -- Cancellations are irreversible
- **R6: Appropriate escalation** -- Transferred with clear summary
- **R23: Knowing when to transfer**

---

#### Task 53: Damaged Bicycle Return
**Scenario:** Return damaged bicycle, refund to original credit card.

**What the model did right:**
1. Found the bicycle order across 3 orders
2. Processed return with credit card payment

**Skills demonstrated:**
- **R5: Cross-order item identification**
- **R6: Return execution**
- **R4: Correct payment method selection**

---

#### Task 55: Financial Emergency -- Cancel/Return All Possible Orders
**Scenario:** Cancel all pending orders, return all delivered orders. 6 orders total.

**What the model did right:**
1. Retrieved all 6 orders
2. Identified 2 pending (cancellable) and 2 delivered (returnable), others already cancelled/processed
3. Cancelled both pending orders
4. Returned items from both delivered orders
5. Managed emotional, stressed user throughout

**Skills demonstrated:**
- **R16: Order status awareness** -- Correctly categorized pending vs. delivered vs. already-cancelled
- **R15: Sequential multi-task handling** -- 4 operations across 4 orders
- **R8: Difficult customer handling** -- Managed stressed, emotional user

---

#### Task 58: Multi-Item Exchange with Preference Cascade (Coffee Machine + Laptop)
**Scenario:** Exchange coffee machine (lower pressure: 8 bar > 9 bar > 7 bar) and laptop (cheapest i7+). Gift card for payment, otherwise credit card.

**What the model did right:**
1. Retrieved both product catalogs
2. Applied pressure preference cascade: 8 bar unavailable, 9 bar found
3. Found cheapest i7+ laptop option
4. Applied payment preference (gift card first)
5. Executed multi-item exchange in single call

**Skills demonstrated:**
- **R1: Product catalog navigation** -- Multi-attribute search
- **R2: Fallback logic** -- Preference cascade (8 > 9 > 7 bar)
- **R3: Multi-item exchange execution**
- **R4: Correct payment method selection** -- Applied preference ordering

---

#### Task 61: Item Modification with Price Constraint
**Scenario:** Change wireless earbuds to blue, price must be same or lower.

**What the model did right:**
1. Found blue earbuds option
2. Verified price was lower ($226.49 vs $256.67)
3. Modified pending order

**Skills demonstrated:**
- **R1: Product catalog navigation**
- **R19: Price-based comparison** -- Verified price constraint
- **R11: Pending order modification**

---

#### Task 63: Off-Topic Request + Price Check + Conditional Item Swap
**Scenario:** User first asks agent to guess a poem (off-topic), then checks speaker price, conditionally cancels if >$300, finds cheaper alternative, updates order.

**What the model did right:**
1. Handled off-topic request appropriately (declined poem guessing)
2. Identified speaker price ($302.67) and battery life (20 hours)
3. Found cheaper speaker alternatives (<$300)
4. Swapped item and calculated new total ($1288.65)

**Skills demonstrated:**
- **R24: Off-topic request handling** -- Redirected to actual task
- **R9: Information communication** -- Price, battery life, new total
- **R19: Price-based comparison**
- **R20: Guided decision tree navigation** -- Followed user's conditional logic

---

#### Task 68: Info Lookup with Zip Code Correction
**Scenario:** Check total of most recent order. User gives wrong zip code first, then corrects.

**What the model did right:**
1. Handled initial failed lookup gracefully
2. Accepted corrected zip code and found user
3. Identified most recent order and communicated total ($829.43)

**Skills demonstrated:**
- **R25: Error recovery from user input** -- Handled wrong zip code
- **R9: Information communication**
- **R26: Recency identification** -- Found most recent order

---

#### Task 69: Return-to-Cancel Pivot
**Scenario:** Return delivered laptop. If can't return, cancel. Order turns out to be pending (not delivered), so cancel instead.

**What the model did right:**
1. Found the order was actually pending, not delivered
2. Recognized return not applicable to pending order
3. Pivoted to cancellation as user's alternative

**Skills demonstrated:**
- **R16: Order status awareness** -- Correctly identified pending status
- **R13: Adaptive intent tracking** -- Switched from return to cancel per user's fallback
- **R27: Cancel execution**

---

#### Task 70: Helmet Exchange with Multi-Attribute Preference
**Scenario:** Exchange helmet: medium size, high ventilation, blue preferred. Communicate price difference ($22.55).

**What the model did right:**
1. Found correct variant (medium, high ventilation, blue)
2. Calculated and communicated price difference
3. Executed exchange with original payment method

**Skills demonstrated:**
- **R1: Product catalog navigation**
- **R9: Information communication** -- "$22.55"
- **R3: Exchange execution**

---

#### Task 73: Return Everything Except One Item
**Scenario:** Return all items from order except coffee machine.

**What the model did right:**
1. Retrieved order and identified all items
2. Excluded coffee machine from return
3. Returned remaining 4 items in single API call

**Skills demonstrated:**
- **R14: Selective item identification** -- Excluded specified item
- **R6: Multi-item return execution**

---

#### Task 75: Direct Exchange with Exact Specs
**Scenario:** Exchange earbuds from blue/8hr/IPX4 to black/4hr/not resistant.

**What the model did right:**
1. Found exact matching variant in product catalog
2. Executed exchange cleanly

**Skills demonstrated:**
- **R1: Product catalog navigation** -- Exact attribute matching
- **R3: Exchange execution**

---

#### Task 77: Exchange for Largest Size (Perfume)
**Scenario:** User loves perfume, wants to exchange for the maximum size available.

**What the model did right:**
1. Retrieved product catalog
2. Found largest available size
3. Executed exchange

**Skills demonstrated:**
- **R1: Product catalog navigation** -- Max-value attribute search
- **R3: Exchange execution**

---

#### Task 78: Triple Operation (Address Change + Item Exchange + Order Cancel)
**Scenario:** Change address on order to match another order, exchange makeup kit, cancel third order.

**What the model did right:**
1. Retrieved all three orders
2. Copied address from one order to another
3. Found matching makeup kit variant (dark, Brand A)
4. Cancelled third order
5. Executed all three operations

**Skills demonstrated:**
- **R12: Address modification** -- Cross-order address copy
- **R1: Product catalog navigation**
- **R17: Mixed operation execution** -- 3 different operations in one session
- **R27: Cancel execution**

---

#### Task 81: Cancel Multiple Orders (Life Changes)
**Scenario:** Cancel orders containing hiking boots, watch, keyboard, charger, jacket, running shoes. If can't cancel items individually, cancel whole orders.

**What the model did right:**
1. Found all items across multiple orders
2. Recognized partial cancellation is not possible
3. Cancelled both entire pending orders

**Skills demonstrated:**
- **R7: Policy constraint recognition** -- Partial item cancellation not allowed
- **R5: Cross-order item identification**
- **R27: Cancel execution** -- Multiple orders

---

#### Task 84: Return with Mind Change (Expensive vs. Cheap Tablet + Payment Switch)
**Scenario:** Initially wants to return cheaper tablet to credit card. Changes mind: return expensive tablet to gift card.

**What the model did right:**
1. Identified both tablets and their prices
2. Adapted when user changed from return-cheaper to return-expensive
3. Changed payment method from credit card to gift card
4. Executed correct return

**Skills demonstrated:**
- **R13: Adaptive intent tracking** -- Handled mid-conversation pivot
- **R19: Price-based comparison**
- **R4: Correct payment method selection** -- Adapted to changed preference

---

#### Task 85: Modify Pending Order (Fleece Jacket)
**Scenario:** Exchange fleece jacket for large, red, half-zipper variant.

**What the model did right:**
1. Found the order with the jacket
2. Looked up product variants
3. Found matching variant and modified

**Skills demonstrated:**
- **R1: Product catalog navigation**
- **R11: Pending order modification**

---

#### Task 86: Modify Item + Change Default Address (from Order Data)
**Scenario:** Exchange fleece jacket to red/half-zipper AND change default address to Washington DC address found in one of the orders.

**What the model did right:**
1. Retrieved all orders to find the DC address
2. Extracted address from order data
3. Updated default address
4. Modified jacket item

**Skills demonstrated:**
- **R28: Cross-reference data extraction** -- Extracted address from order to update profile
- **R1: Product catalog navigation**
- **R12: Address modification**

---

#### Task 87: Batch Address Update (3 Orders + Default)
**Scenario:** Change all pending order addresses + default address to DC address (from another order).

**What the model did right:**
1. Retrieved all orders, identified the DC address
2. Identified all 3 pending orders
3. Updated all 3 order addresses + user default in one batch
4. Used address data from another order

**Skills demonstrated:**
- **R28: Cross-reference data extraction** -- Extracted address from order
- **R12: Address modification** -- 4 simultaneous address updates
- **R16: Order status awareness** -- Only updated pending orders

---

#### Task 88: Conditional Cancel (Unavailable Variant)
**Scenario:** Change bookshelf to 4-foot. If unavailable in same material/color, cancel the whole order.

**What the model did right:**
1. Checked product variants for 4-foot option
2. Found no matching variant (same material and color)
3. Cancelled whole order as per fallback instruction

**Skills demonstrated:**
- **R1: Product catalog navigation**
- **R2: Fallback logic** -- Triggered cancellation when preferred option unavailable
- **R27: Cancel execution**

---

#### Task 89: Conditional Return vs. Exchange (Price-Based)
**Scenario:** Find cheapest mechanical keyboard. If <$200, exchange. If not, return current one.

**What the model did right:**
1. Found cheapest keyboard ($226.11) and communicated its attributes (tactile, white, full)
2. Since $226.11 > $200, processed return instead of exchange

**Skills demonstrated:**
- **R1: Product catalog navigation** -- Min-value search
- **R9: Information communication** -- Price and attributes
- **R20: Guided decision tree navigation** -- Applied price-based conditional logic

---

#### Task 94: Laptop Exchange (Specific Specs)
**Scenario:** Exchange laptop to i7, 8GB, 1TB SSD.

**What the model did right:**
1. Found the laptop order
2. Matched specs in product catalog
3. Executed exchange

**Skills demonstrated:**
- **R1: Product catalog navigation**
- **R3: Exchange execution**

---

#### Task 95: Dual Laptop Exchange Across Orders + Price Communication
**Scenario:** Exchange two 15-inch laptops from different orders to i7/8GB/1TB. Communicate total price difference ($167.87) broken into per-order amounts ($60.78 + $107.09).

**What the model did right:**
1. Found both laptop orders
2. Matched same target spec for both
3. Executed both exchanges
4. Calculated and communicated individual and total price differences

**Skills demonstrated:**
- **R1: Product catalog navigation**
- **R3: Exchange execution** -- Dual exchange
- **R9: Information communication** -- 3 price values
- **R7: Correct policy math** -- Price difference calculation

---

#### Task 96: Address Change + Item Modification (Cross-Reference)
**Scenario:** Change LA order address to NYC address (from another order). Exchange bluetooth speaker to cheapest green variant.

**What the model did right:**
1. Found NYC address from another order
2. Updated LA order's address
3. Found cheapest green speaker variant
4. Modified item

**Skills demonstrated:**
- **R28: Cross-reference data extraction**
- **R12: Address modification**
- **R1: Product catalog navigation** -- Min-price with color constraint
- **R11: Pending order modification**

---

#### Task 98: Complex Multi-Operation (2 Exchanges + 1 Cancel + Payment Preference)
**Scenario:** Exchange bicycle (larger frame) + jigsaw puzzle (1000 more pieces, animal theme preferred) + exchange camera (lower resolution) + cancel skateboard. Different payment preference.

**What the model did right:**
1. Navigated 3 separate product catalogs
2. Applied preference ordering (animal > art theme)
3. Executed 2 exchanges and 1 cancellation
4. Applied alternative payment method per user's late preference change

**Skills demonstrated:**
- **R1: Product catalog navigation** -- Multiple products
- **R2: Fallback logic** -- Theme preference
- **R17: Mixed operation execution** -- 3 distinct operations
- **R4: Correct payment method selection** -- Late preference change
- **R13: Adaptive intent tracking** -- Payment method pivot at confirmation

---

#### Task 105: Exchange Two Identical Items to Different Variants
**Scenario:** Exchange two identical tea kettles to different target variants (one to ceramic/gas, other to 1.5L/gas).

**What the model did right:**
1. Identified both identical items in the order
2. Found correct target variants for each
3. Presented exchange details clearly

**Skills demonstrated:**
- **R29: Duplicate item differentiation** -- Handled two identical items with different target variants
- **R1: Product catalog navigation**

**Note:** Verified that no `exchange_delivered_order_items` call was made in this trace -- the user backed out due to payment concerns. The task passed via the NL assertion evaluator ("Agent should exchange both tea kettles to the items requested"), which likely assessed that the agent correctly identified both items and presented the correct exchange plan, even though the user ultimately declined. This is a case where the DB check passed trivially (no mutations expected since the user declined) and the NL assertion evaluated the agent's demonstrated *capability* rather than *execution*.

---

#### Task 108: Return All Except Tablet + Refund Communication
**Scenario:** Return everything except tablet from delivered order. Communicate refund amount ($346.93).

**What the model did right:**
1. Found the correct order
2. Excluded tablet from return
3. Processed return for 3 items
4. Communicated refund amount

**Skills demonstrated:**
- **R14: Selective item identification**
- **R6: Return execution**
- **R9: Information communication** -- "$346.93"

---

#### Task 113: Cancel All Pending Orders
**Scenario:** Cancel all pending orders. Don't reveal reason until asked.

**What the model did right:**
1. Retrieved all orders
2. Identified the 2 pending orders
3. Asked for cancellation reason
4. Cancelled both with "ordered by mistake"

**Skills demonstrated:**
- **R16: Order status awareness** -- Correctly filtered pending orders
- **R27: Cancel execution** -- Multiple orders
- **R30: Information elicitation** -- Asked for required reason before proceeding

---

## 3. CORE SKILL TAXONOMY

The previous section documented what happened in each task individually. This section distills all 54 successes into **7 core skills**---concrete, observable capabilities that the model demonstrated. Every skill is grounded in specific behavioral evidence from the conversation traces, and every success maps to at least one skill. These are not abstract categories---each one corresponds to a specific thing the model *did* that directly caused the task to succeed.

### How We Derived These Skills

We analyzed all 54 successful conversations extracting:
- Every tool call sequence and its arguments
- Every agent decision point (what the agent said at critical moments)
- The evaluation criteria that determined success (DB checks, NL assertions, communicate_info)
- The contrast against the 96 failures to confirm that skill absence---not task randomness---explains failure

The 7 skills below collectively explain 100% of successes. Each task requires 1-4 of these skills; no task requires all 7.

---

### Skill 1: User Identity Resolution
**What it is:** Authenticating the user and resolving their identity through available methods (name+zip, email, user ID), including recovering from errors (wrong zip code, invalid email).

**Observable behavior:**
- Calling `find_user_id_by_name_zip` or `find_user_id_by_email` with correct parameters extracted from conversation
- Calling `get_user_details` to retrieve the full user profile
- Retrying with corrected input when first attempt fails

**Evidence from successes:**
- 53 of 54 successes begin with successful user authentication (Task 105 is the exception; user provided order ID directly)
- Task 68 (retail): User gave wrong zip code (98178), agent recovered with correct one (98187)
- Task 35 (retail): Agent handled dual email authentication (two emails for the same account)
- Task 1 (airline): Agent retried `get_user_details` after initial wrong user ID

**Why it matters:** Authentication is the gateway to all subsequent actions. The model never attempted a mutation without first confirming who the user was. In the 12 failures where no mutation was attempted (retail), several trace back to authentication deadlocks.

| Domain | Tasks requiring this skill |
|--------|--------------------------|
| Airline | All 12 (0,1,3,4,5,6,28,34,40,45,46,48) |
| Retail | 41 of 42 (all except 105) |

---

### Skill 2: Systematic Data Retrieval
**What it is:** Looking up all relevant records (reservations, orders, products, flight statuses) *before* making any decision or taking any action. The model retrieves first and reasons second---never the reverse.

**Observable behavior:**
- Calling `get_reservation_details`/`get_order_details` for all relevant records
- Calling `get_product_details` to enumerate available variants before proposing an exchange or modification
- Calling `get_flight_status` to verify actual flight conditions
- Searching across multiple records when the target record is not known upfront

**Evidence from successes:**
- Task 4 (airline): Agent checked ALL 5 reservations and ALL flight statuses (15 lookups) to disprove user's fabricated cancellation story
- Task 1 (airline): Agent searched through 5 reservations to find the PHL-LGA trip the user mentioned but didn't provide an ID for
- Task 55 (retail): Agent looked up all 6 orders to identify which were pending vs. delivered
- Task 87 (retail): Agent looked up 5 orders to find which contained the Washington DC address and which were pending

**Quantitative pattern:**
- Airline successes averaged **4.1 lookup calls** per task
- Retail successes averaged **3.0 lookup calls** per task
- More lookups correlated with higher task complexity, not lower success---the model consistently invested in data gathering

| Domain | Tasks requiring this skill |
|--------|--------------------------|
| Airline | All 12 |
| Retail | All 42 |

---

### Skill 3: Policy-Grounded Refusal Under Pressure
**What it is:** Correctly determining that a user's request violates policy, explicitly refusing to comply, and maintaining that refusal across multiple rounds of social pressure including deception, emotional manipulation, authority claims, and aggressive negotiation.

**Observable behavior:**
- Enumerating specific policy conditions and checking each one against retrieved data
- Saying "no" with a clear policy rationale
- Repeating the refusal when the user escalates, changes tactic, or makes unverifiable claims
- Escalating to human agents (with a clear summary) rather than caving

**Evidence from successes:**
- Task 0 (airline): Enumerated all 4 cancellation conditions, denied, resisted "I was told I didn't need insurance"
- Task 1 (airline): Denied despite user's "a previous agent approved it" claim, escalated
- Task 28 (airline): Refused gift card, voucher, 50% refund, and 10% refund offers---zero unauthorized concessions
- Task 45 (airline): Denied cancellation across 3+ rounds despite family emergency emotional pressure
- Task 48 (airline): Contradicted user's "booked 10 hours ago" claim with system timestamp showing 13 days

**User pressure tactics successfully resisted:**

| Tactic | Tasks |
|--------|-------|
| False membership/status claim | 3, 4, 5 |
| False prior approval | 1 |
| False timeline | 48 |
| Fabricated events | 4 |
| Emotional manipulation (emergency, urgency) | 0, 45 |
| Aggressive negotiation (partial refund demands) | 28 |
| Persistent repetition | 6, 46 |

**This skill is the dominant airline skill.** 11 of 12 airline successes (all except Task 40) required it. 10 of these 11 had explicit refusal NL assertions ("Agent should NOT [do X]"). Task 3 required rejecting a false Gold membership claim (a refusal-adjacent verification skill---the agent must resist accepting the user's claimed status and communicate the corrected facts).

**Caveats:** In 2 cases (Tasks 5 and 6), the agent's reasoning was actually wrong---it verbally accepted false claims or tried to execute disallowed operations---but the outcome was correct because either the conversation ended before execution (Task 5) or the API rejected the invalid operation (Task 6). True clean refusals: 9/11.

| Domain | Tasks requiring this skill |
|--------|--------------------------|
| Airline | 11 (0,1,3,4,5,6,28,34,45,46,48) |
| Retail | 5 (11,38,50,69,81---policy constraint recognition, e.g., "partial cancellation not possible") |

---

### Skill 4: Product/Variant Matching
**What it is:** Navigating a product catalog to find the correct item variant given a set of desired attributes, handling preference orderings (first choice > fallback), and identifying the right `item_id` / `new_item_id` for the API call.

**Observable behavior:**
- Calling `get_product_details` and scanning the returned variant list
- Matching on required attributes while being flexible on optional ones
- Applying preference cascades (e.g., "8 bar, else 9 bar, else 7 bar")
- Finding min/max priced variants when requested

**Evidence from successes:**
- Task 0 (retail): Searched keyboard variants for clicky+full-size+RGB; RGB unavailable, correctly fell back to no-backlight
- Task 58 (retail): Applied pressure cascade for coffee machine (8 bar unavailable → 9 bar found), found cheapest i7+ laptop
- Task 89 (retail): Found cheapest keyboard ($226.11 > $200 threshold → correctly chose return over exchange)
- Task 35 (retail): Found 13-inch laptop matching i5 > i7 preference and silver/black color preference
- Task 98 (retail): Navigated 3 separate product catalogs (bicycle, puzzle, camera) with attribute constraints

**This is where failures diverge most clearly from successes.** In the retail failure taxonomy:
- **48 failures** had wrong `item_ids` (picked wrong item from order)
- **34 failures** had wrong `new_item_ids` (picked wrong variant from product catalog)

The successes got variant matching right; the failures got it wrong. Same skill, different outcomes.

| Domain | Tasks requiring this skill |
|--------|--------------------------|
| Airline | 0 (airline has no product catalog) |
| Retail | 22 (0,2,15,18,35,58,61,63,70,75,77,78,84,85,86,88,89,94,95,96,98,105) |

---

### Skill 5: Correct State-Mutating API Execution
**What it is:** Calling the right mutation function (`return_delivered_order_items`, `exchange_delivered_order_items`, `cancel_pending_order`, `modify_pending_order_*`, `update_reservation_passengers`) with exactly the right arguments---correct `order_id`, correct `item_ids`, correct `new_item_ids`, correct `payment_method_id`.

**Observable behavior:**
- Choosing the right mutation function for the operation type AND order status (return/exchange for delivered; modify/cancel for pending)
- Passing complete, correct argument lists
- Getting user confirmation before executing
- Successfully executing the call (no API errors on the final attempt)

**Evidence from successes:**
- 37 of 42 retail successes had **exact argument matches** against the expected mutations (88.1%). The other 5 succeeded via: item_id ordering differences (Tasks 73, 108), optional transfer not executed (Task 26), transfer summary wording (Task 50), and NL assertion pass without execution (Task 105).
- Task 40 (airline): Correctly passed the FULL passenger list (both passengers, not just the changed one) to `update_reservation_passengers`
- Task 78 (retail): Executed 3 different mutation types in one session---address change, item modification, and order cancellation---all with correct arguments
- Task 87 (retail): Executed 4 address updates (3 orders + user default) all with correct address data extracted from another order

**Contrast with failures:** This is precisely what failures got wrong:
- 48 failures: wrong `item_ids` (wrong item selected from order)
- 34 failures: wrong `new_item_ids` (wrong product variant)
- 36 failures: wrong `order_id` (operated on wrong order)
- 13 failures: wrong `payment_method_id`
- 25 failures: called the wrong function entirely (e.g., exchange instead of return)

| Domain | Tasks requiring this skill |
|--------|--------------------------|
| Airline | 1 (Task 40 only) |
| Retail | 40 (all except Tasks 68, 105 which required no mutations) |

---

### Skill 6: Information Communication
**What it is:** Extracting specific data values from tool results (prices, tracking numbers, account balances, product specs) and relaying them to the user accurately, in the format they requested.

**Observable behavior:**
- Reading a specific field from a tool call result (e.g., order total, tracking number, gift card balance)
- Computing derived values (e.g., price difference = new_price - old_price)
- Communicating the value in the response text where the user can see it

**Evidence from successes:**
- Task 3 (airline): Communicated "4" (total suitcases) after computing Silver + economy = 2 bags/person × 2 passengers
- Task 43 (retail): Communicated 7 separate values: tracking number, full address (5 components), and tablet storage (64GB)
- Task 95 (retail): Communicated 3 price differences: $60.78, $107.09, and total $167.87
- Task 38 (retail): Communicated "camera" as most expensive item and "$481.50" as its price
- Task 63 (retail): Communicated speaker price ($302.67), battery life (20 hours), and new order total ($1288.65)

**Difficulty signal:** Tasks with `communicate_info` requirements succeed at 28.9% vs. 40.8% without (retail). This skill is materially harder than pure execution---the model must both DO the right thing and SAY the right thing.

| Domain | Tasks requiring this skill |
|--------|--------------------------|
| Airline | 1 (Task 3: baggage count) |
| Retail | 11 (2,32,38,40,43,63,68,70,89,95,108) |

---

### Skill 7: Multi-Step Task Orchestration
**What it is:** Managing a conversation that requires multiple distinct operations---potentially across different orders, different operation types (return + cancel + exchange), and interleaved with user requests that arrive sequentially or with mid-conversation pivots.

**Observable behavior:**
- Tracking which operations have been completed and which remain
- Handling user mind-changes mid-conversation (e.g., switch from return to exchange)
- Operating across multiple orders in a single session
- Adapting when an earlier step's outcome changes the plan for later steps

**Evidence from successes:**
- Task 32 (retail): 4 operations across 4 orders, revealed sequentially by user: tracking lookup → cancel charger → cancel boot+kettle → return sneaker (44 messages)
- Task 55 (retail): 4 operations across 4 orders: cancel 2 pending + return 2 delivered (34 messages)
- Task 98 (retail): 3 operations across 3 orders: exchange bicycle + exchange puzzle+camera + cancel skateboard
- Task 78 (retail): 3 different operation types in one session: address change + item modification + order cancel
- Task 84 (retail): User initially wanted to return cheaper tablet to credit card, mid-conversation pivoted to returning expensive tablet to gift card---agent tracked the change correctly

**Complexity correlation:** Tasks with ≥2 complexity signals (multi-order, multi-mutation, mixed-operations, conditional logic, etc.) succeed at 31.9% vs. 42.9% for tasks with 0-1 signals. The average successful task has 1.62 complexity signals vs. 2.21 for failures.

| Domain | Tasks requiring this skill |
|--------|--------------------------|
| Airline | 1 (Task 34: multiple flight searches + cost calculation) |
| Retail | 19 (2,11,13,14,26,32,35,48,55,78,81,85,86,87,95,96,98,113, and mind-change tasks 18,69,84) |

---

### Skill-to-Task Mapping Matrix

Each cell indicates whether the skill was **required** for that task's success.

#### Airline (12 successes)

| Task | S1:Identity | S2:Data | S3:Refusal | S4:Variant | S5:Mutation | S6:Comms | S7:Orchestration |
|------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 0  | x | x | x | - | - | - | - |
| 1  | x | x | x | - | - | - | - |
| 3  | x | x | x | - | - | x | - |
| 4  | x | x | x | - | - | - | - |
| 5  | x | x | x | - | - | - | - |
| 6  | x | x | x | - | - | - | - |
| 28 | x | x | x | - | - | - | - |
| 34 | x | x | x | - | - | - | x |
| 40 | x | x | - | - | x | - | - |
| 45 | x | x | x | - | - | - | - |
| 46 | x | x | x | - | - | - | - |
| 48 | x | x | x | - | - | - | - |

#### Retail (42 successes)

| Task | S1:Identity | S2:Data | S3:Refusal | S4:Variant | S5:Mutation | S6:Comms | S7:Orchestration |
|------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 0  | x | x | - | x | x | - | - |
| 2  | x | x | - | x | x | x | x |
| 11 | x | x | x | - | x | - | x |
| 13 | x | x | - | - | x | - | x |
| 14 | x | x | - | - | x | - | x |
| 15 | x | x | - | x | x | - | - |
| 17 | x | x | - | - | x | - | - |
| 18 | x | x | - | x | x | - | x |
| 26 | x | x | - | - | x | - | x |
| 32 | x | x | - | - | x | x | x |
| 35 | x | x | - | x | x | - | x |
| 38 | x | x | x | - | x | x | - |
| 40 | x | x | - | - | x | x | - |
| 43 | x | x | - | - | x | x | - |
| 48 | x | x | - | - | x | - | x |
| 50 | x | x | x | - | x | - | - |
| 53 | x | x | - | - | x | - | - |
| 55 | x | x | - | - | x | - | x |
| 58 | x | x | - | x | x | - | - |
| 61 | x | x | - | x | x | - | - |
| 63 | x | x | - | x | x | x | - |
| 68 | x | x | - | - | - | x | - |
| 69 | x | x | x | - | x | - | x |
| 70 | x | x | - | x | x | x | - |
| 73 | x | x | - | - | x | - | - |
| 75 | x | x | - | x | x | - | - |
| 77 | x | x | - | x | x | - | - |
| 78 | x | x | - | x | x | - | x |
| 81 | x | x | x | - | x | - | x |
| 84 | x | x | - | x | x | - | x |
| 85 | x | x | - | x | x | - | x |
| 86 | x | x | - | x | x | - | x |
| 87 | x | x | - | - | x | - | x |
| 88 | x | x | - | x | x | - | - |
| 89 | x | x | - | x | x | x | - |
| 94 | x | x | - | x | x | - | - |
| 95 | x | x | - | x | x | x | x |
| 96 | x | x | - | x | x | - | x |
| 98 | x | x | - | x | x | - | x |
| 105| x | x | - | x | - | - | - |
| 108| x | x | - | - | x | x | - |
| 113| x | x | - | - | x | - | x |

#### Skill Frequency Summary

| Skill | Airline (of 12) | Retail (of 42) | Total (of 54) |
|-------|:-:|:-:|:-:|
| S1: User Identity Resolution | 12 (100%) | 41 (98%) | 53 (98%) |
| S2: Systematic Data Retrieval | 12 (100%) | 42 (100%) | 54 (100%) |
| S3: Policy-Grounded Refusal | 11 (92%) | 5 (12%) | 16 (30%) |
| S4: Product/Variant Matching | 0 (0%) | 22 (52%) | 22 (41%) |
| S5: Correct Mutation Execution | 1 (8%) | 40 (95%) | 41 (76%) |
| S6: Information Communication | 1 (8%) | 11 (26%) | 12 (22%) |
| S7: Multi-Step Orchestration | 1 (8%) | 19 (45%) | 20 (37%) |

---

## 4. KEY FINDINGS AND PATTERNS

### Finding 1: Two Universal Skills, Five Specialized Ones
**S1 (Identity Resolution) and S2 (Data Retrieval) are required for virtually every task.** They are table stakes. The model does these well and consistently. The remaining 5 skills are specialized---a task requires some subset of them, and the model's success depends on which combination is needed and at what difficulty level.

### Finding 2: The Airline-Retail Skill Profile is Strikingly Asymmetric
- **Airline successes are almost entirely S3 (Refusal):** 11/12 airline successes required S3, and only 1 required S5 (Mutation Execution). The model's airline competence is saying "no."
- **Retail successes are almost entirely S5 (Mutation Execution):** 40/42 retail successes required S5, and only 5 required S3. The model's retail competence is doing things.
- **The model lacks overlap:** It is good at refusal OR execution, rarely both in complex combination.

### Finding 3: S4 (Variant Matching) is the Key Differentiator Between Retail Success and Failure
- **48 retail failures had wrong `item_ids`; 34 had wrong `new_item_ids`.** These are exactly the parameters that S4 produces.
- 22 of 42 retail successes required S4 and got it right. When the model correctly identifies the product variant, the rest of the execution pipeline (S5) succeeds almost automatically.
- Success rate for tasks requiring S4: 22/(22+~30 failures needing it) ≈ 42%. Success rate for tasks NOT requiring variant matching: ~50%. Variant matching is a bottleneck.

### Finding 4: Complexity is the Primary Difficulty Driver (Quantified)
- Average complexity signals per successful task: **1.62**
- Average complexity signals per failed task: **2.21**
- Success rate by complexity:

| Complexity Signals | Retail Success Rate |
|:-:|:-:|
| 0 | 50.0% |
| 1 | 42.1% |
| 2 | 33.3% |
| 3+ | 23.5% |

Complexity signals: conditional_logic, multi_order_search, multi_product_lookup, multi_mutation, mixed_operations, info_communication, user_mind_change, information_withholding.

### Finding 5: S3 (Refusal) Has a 2-Task Asterisk
In Tasks 5 and 6 (airline), the agent's *reasoning* was wrong (verbally accepted false Gold claim; tried to add insurance via API), but the *outcome* was correct because of external factors (conversation ended before execution; API rejected the operation). True clean S3 demonstrations: **9/11 airline, not 11/11.** This suggests the model's refusal skill is strong but not perfectly reliable---it sometimes caves to pressure at the reasoning level, getting saved by execution-level safeguards.

### Finding 6: S6 (Communication) is Disproportionately Hard
- Tasks with `communicate_info` requirements: **28.9% success rate**
- Tasks without: **40.8% success rate**
- The model sometimes does the right thing (correct mutation) but fails to SAY the right thing (omits a price or tracking number). Communication requires extracting specific values from tool results and surfacing them---a different skill from pure action execution.

### Finding 7: S7 (Orchestration) Scales Poorly
- Single-operation tasks: ~45% success rate
- Multi-operation tasks (≥2 mutations): ~32% success rate
- Tasks with ≥3 expected mutations: ~27% success rate
- The model handles individual operations well but loses coherence when juggling multiple operations across orders. The 19 successful multi-step tasks represent its ceiling---not a comfortable operating range.

---

## 5. SKILL DIFFICULTY HIERARCHY

### Reliable (Model demonstrates consistently):
1. **S1: Identity Resolution** -- Authentication succeeds in 98% of successes
2. **S2: Data Retrieval** -- The model always looks before it leaps

### Strong (Model demonstrates in most applicable tasks):
3. **S3: Policy-Grounded Refusal** -- 9 of 11 clean demonstrations; strong but with 2 asterisks
4. **S5: Correct Mutation Execution** -- 88.1% exact argument match in successes

### Moderate (Model demonstrates in roughly half of applicable tasks):
5. **S4: Product/Variant Matching** -- ~42% success rate when required; the key bottleneck
6. **S7: Multi-Step Orchestration** -- ~32% success rate for multi-operation tasks

### Weak (Model frequently fails):
7. **S6: Information Communication** -- 28.9% success when specific values must be communicated

---

## 6. FAILURE CONTRAST: What Skill Gaps Explain the 96 Failures?

### Airline Failures (38/50): Absent S5 (Mutation Execution)

The model's airline profile is lopsided: **S3 (Refusal) is strong; S5 (Mutation Execution) is almost entirely absent.**

| Failure Category | Count | Missing Skill |
|-----------------|:-----:|:--------------|
| Flight modification (change, upgrade, downgrade) | 14 | S5: Required computing price differences and executing `update_reservation_flights` with correct args |
| Cancellation where cancellation IS allowed | 11 | S5: Required calling `cancel_reservation` correctly + S3 in reverse (knowing WHEN to act) |
| Flight booking | 4 | S5: Required `book_reservation` with exact pricing, S4 (flight selection = variant matching) |
| Search/info | 3 | S6: Required multi-step computation (cheapest routes, durations) |
| Complex multi-operation | 3 | S7: Required sequencing multiple operations |
| max_steps timeout | 3 | S7: Agent couldn't converge within turn limit |

**Key insight:** 11/12 successes required only S3 (refuse the disallowed operation). 35/38 failures required S5 (execute the correct operation). The model can judge that something is NOT allowed; it cannot reliably construct the correct parameters for something that IS allowed.

### Retail Failures (72/114): Granular Argument Errors in S4 and S5

Retail failures were analyzed by examining the exact argument mismatches between expected and actual mutation calls:

| Error Type | Occurrences | Skill Gap |
|-----------|:-----------:|:----------|
| Wrong `item_ids` (picked wrong item from order) | 48 | S2/S5: Misidentified which item the user meant |
| Wrong `order_id` (operated on wrong order) | 36 | S2: Failed to identify the correct order |
| Wrong `new_item_ids` (picked wrong product variant) | 34 | S4: Selected wrong variant from catalog |
| Wrong function type (e.g., exchange vs return) | 25 | S5: Confused operation types |
| Wrong mutation count (too many or too few) | 15 | S7: Lost track of how many operations needed |
| No mutation attempted at all | 15 | S5: Complete execution failure |
| Wrong `payment_method_id` | 13 | S5: Used wrong payment method |
| Wrong address fields | 6 | S5: Address copy errors |

**The #1 retail failure mode is wrong `item_ids` (48 occurrences).** This means the agent looked at an order, misidentified which item the user was referring to, and operated on the wrong one. This is a failure at the intersection of S2 (data retrieval interpretation) and S5 (mutation parameterization).

**The #2 failure mode is wrong `order_id` (36 occurrences).** The agent operated on the wrong order entirely---a failure of S2 (searching across multiple orders to find the right one).

### What Made Successes Succeed Where Failures Failed

Mapping failures to the skill taxonomy reveals that **every failure can be attributed to a specific skill gap:**

1. **S4 gap (variant matching):** 34 failures selected the wrong product variant. The 22 successes requiring S4 got the variant right---same task structure, different outcome on variant selection.

2. **S5 gap (mutation execution):** 25 failures called the wrong function entirely. The 40 successes requiring S5 called the right function with 88.1% exact argument accuracy.

3. **S7 gap (orchestration):** 15 failures had the wrong number of mutations. The 19 successes requiring S7 tracked all operations correctly (average 2.7 mutations each).

4. **S2 gap (data retrieval):** 36 failures operated on the wrong order. Successes that involved multi-order search (13 tasks) correctly identified the right order every time.

The model does not fail randomly. It fails when a specific skill is pushed beyond its reliability threshold---and that threshold is lower for S4 (variant matching) and S7 (orchestration) than for S1-S3.
