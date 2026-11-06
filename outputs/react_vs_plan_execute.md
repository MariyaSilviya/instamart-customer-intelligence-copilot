# ReAct vs Plan-Execute

## ReAct

ReAct combines reasoning and action in an iterative loop. The agent
selects a tool, receives its observation, and decides the next action.
This is useful for the refund assistant because the next action depends
on the order lookup and policy result. For example, the assistant must
check order status before calculating a refund and must escalate refunds
above INR 1000.

## Plan-Execute

Plan-Execute first creates a sequence of tasks and then executes that
plan. It is useful when a workflow has predictable stages, such as
loading customer information, checking eligibility, calculating an
amount, and preparing a report. It can make execution easier to inspect.

## Comparison

ReAct is more adaptive when tool results change the next step. Plan-Execute
is more structured and can reduce repeated planning for predictable work.
Both need tool validation, iteration limits, token limits, timeouts,
checkpointing, and human escalation. For this refund workflow, ReAct is
appropriate because order status and refund eligibility determine which
tool should run next.
