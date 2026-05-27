import Anthropic from "@anthropic-ai/sdk";
import type { InboxItem, ItemOutput } from "./types.js";
import {
  create_task,
  draft_message,
  escalate,
  find_slots,
  getToolCallsForItem,
  hold_slot,
  lookup_policy,
  search_patient,
  verify_insurance,
  withItemContext,
} from "./tools.js";

const client = new Anthropic();

const SYSTEM_PROMPT = `You are a triage agent for Cedar Kids Therapy, a pediatric therapy practice serving children ages 0-18 for speech-language pathology (SLP), occupational therapy (OT), and physical therapy (PT).

It is Monday 8am. You are processing the weekend inbox. For each item, use the available tools to gather information, then call submit_triage with your final assessment.

## Policies

**Insurance:**
- In-network payers: Aetna, Blue Cross Blue Shield, UnitedHealthcare, Medicaid
- Out-of-network payers: Kaiser, Cigna Select, Beacon — require a benefits conversation before any slot is held
- Verified billing system status supersedes referral documents; surface any discrepancy

**Safeguarding:**
- Any disclosure of harm, abuse, neglect, or unsafe caregiving is P0
- Escalate to clinical lead immediately, create a same-hour review task
- Draft a neutral acknowledgement only — no investigative advice

**Clinical advice:**
- Front desk and automated systems must not provide clinical advice
- Route clinical questions to evaluation, screening, or clinician review

**Scheduling:**
- Same-day cancellations or reschedules are P1 operational issues
- find_slots and hold_slot are for human review only — do not schedule appointments

**Language access:**
- Spanish-speaking families: draft reply in Spanish, set language="es", match with Spanish-capable provider when finding slots

## Urgency Calibration
- P0: safeguarding, imminent harm. Same-hour human review.
- P1: same-day operational issue requiring prompt staff action.
- P2: normal intake, scheduling, billing, or clinical-review workflow. This is the default.
- P3: low-priority admin, FYI, spam.

Default to P2 unless there is a clear safety or same-day operational reason. Over-escalation is itself a production failure mode.

## Tool Usage Guidelines
- verify_insurance: use for any item that includes payer/member_id information
- search_patient: use when you have a name and DOB for a potentially existing patient
- lookup_policy: use when a specific policy area is directly relevant to your decision
- find_slots: use for new referrals with complete intake data that are in-network (or after OON is resolved)
- hold_slot: use after finding slots for a complete, ready-to-schedule referral
- create_task: use to assign actionable follow-up to front_desk, intake, billing, or clinical_lead
- draft_message: use to prepare an outbound communication (never auto-sent)
- escalate: use only for genuine P0/P1 situations

## Draft Message Guidelines
- Clear, empathetic, concise, and operationally useful
- Must not provide clinical advice and must not imply the message was sent
- Spanish items: write the body in Spanish, set language="es"
- channel must be portal, email, or phone — never fax. For fax referrers, draft via email or phone

## submit_triage Field Notes
- task_ids: include the IDs returned by create_task calls (format "task_XXXX")
- escalation: if you called escalate, set { reason, severity } matching what you passed; otherwise null
- tools_called: leave as empty array — it is populated automatically from the audit trace
- draft_reply: copy the body of your draft_message verbatim, or null if no message was drafted`;

const TOOL_DEFINITIONS: Anthropic.Tool[] = [
  {
    name: "search_patient",
    description: "Search for an existing patient record by name and/or date of birth.",
    input_schema: {
      type: "object" as const,
      properties: {
        name: { type: "string", description: "Patient full name" },
        dob: { type: "string", description: "Date of birth in YYYY-MM-DD format" },
      },
    },
  },
  {
    name: "verify_insurance",
    description: "Verify insurance coverage status for a given payer and member ID.",
    input_schema: {
      type: "object" as const,
      properties: {
        payer: { type: "string", description: "Insurance payer name (e.g. 'Blue Cross Blue Shield PPO')" },
        member_id: { type: "string", description: "Insurance member ID" },
      },
    },
  },
  {
    name: "lookup_policy",
    description: "Look up Cedar Kids Therapy policy snippets for a specific topic.",
    input_schema: {
      type: "object" as const,
      properties: {
        topic: {
          type: "string",
          enum: [
            "service_lines",
            "insurance",
            "safeguarding",
            "clinical_advice",
            "scheduling",
            "cancellation",
            "language_access",
          ],
          description: "The policy topic to retrieve",
        },
      },
      required: ["topic"],
    },
  },
  {
    name: "find_slots",
    description: "Find available appointment slots, optionally filtered by discipline, scheduling preferences, or provider language.",
    input_schema: {
      type: "object" as const,
      properties: {
        discipline: {
          type: "string",
          enum: ["SLP", "OT", "PT"],
          description: "Therapy discipline",
        },
        preferences: {
          type: "string",
          description: "Scheduling preferences expressed by the family (e.g. 'after school Tuesdays or Thursdays')",
        },
        language: {
          type: "string",
          description: "Required provider language code (e.g. 'es' for Spanish)",
        },
      },
    },
  },
  {
    name: "hold_slot",
    description: "Place a pending-review hold on a specific slot. Does not schedule — the hold is for human review only.",
    input_schema: {
      type: "object" as const,
      properties: {
        slot_id: { type: "string", description: "Slot ID from find_slots result" },
        patient_ref: { type: "string", description: "Patient name or reference for the hold" },
      },
      required: ["slot_id", "patient_ref"],
    },
  },
  {
    name: "create_task",
    description: "Create a staff task assigned to a specific role.",
    input_schema: {
      type: "object" as const,
      properties: {
        assignee: {
          type: "string",
          enum: ["front_desk", "intake", "billing", "clinical_lead"],
          description: "Staff role responsible for this task",
        },
        title: { type: "string", description: "Short task title" },
        due: { type: "string", description: "Due date in YYYY-MM-DD format" },
        notes: { type: "string", description: "Detailed context for the assignee" },
      },
      required: ["assignee", "title", "due", "notes"],
    },
  },
  {
    name: "draft_message",
    description: "Draft an outbound message to a family or referrer. Does not send — for human review only.",
    input_schema: {
      type: "object" as const,
      properties: {
        recipient: { type: "string", description: "Recipient name or contact" },
        channel: {
          type: "string",
          enum: ["portal", "email", "phone"],
          description: "Communication channel",
        },
        body: { type: "string", description: "Message body text" },
        language: {
          type: "string",
          enum: ["en", "es"],
          description: "Message language (default: en)",
        },
      },
      required: ["recipient", "channel", "body"],
    },
  },
  {
    name: "escalate",
    description: "Escalate an item to clinical lead or operations for urgent same-hour or same-day review.",
    input_schema: {
      type: "object" as const,
      properties: {
        item_id: { type: "string", description: "Inbox item ID being escalated" },
        reason: { type: "string", description: "Clear reason for escalation" },
        severity: {
          type: "string",
          enum: ["P0", "P1"],
          description: "P0 for safeguarding/imminent harm; P1 for same-day operational urgency",
        },
      },
      required: ["item_id", "reason", "severity"],
    },
  },
  {
    name: "submit_triage",
    description: "Submit the completed triage output for this inbox item. Call this once you have finished using tools and are ready to record your assessment.",
    input_schema: {
      type: "object" as const,
      properties: {
        item_id: { type: "string" },
        classification: {
          type: "string",
          enum: [
            "new_referral",
            "existing_patient_request",
            "scheduling",
            "clinical_question",
            "billing_question",
            "missing_paperwork",
            "provider_followup",
            "complaint",
            "safeguarding",
            "spam",
            "other",
          ],
        },
        urgency: { type: "string", enum: ["P0", "P1", "P2", "P3"] },
        requires_human_review: { type: "boolean" },
        extracted_intake: {
          type: "object",
          properties: {
            child_name: { type: ["string", "null"] },
            dob_or_age: { type: ["string", "null"] },
            parent_contact: { type: ["string", "null"] },
            discipline: {
              oneOf: [
                { type: "null" },
                {
                  type: "array",
                  items: { type: "string", enum: ["SLP", "OT", "PT"] },
                  minItems: 1,
                  uniqueItems: true,
                },
              ],
            },
            diagnosis_or_concern: { type: ["string", "null"] },
            payer: { type: ["string", "null"] },
            member_id: { type: ["string", "null"] },
          },
          required: [
            "child_name",
            "dob_or_age",
            "parent_contact",
            "discipline",
            "diagnosis_or_concern",
            "payer",
            "member_id",
          ],
        },
        missing_info: {
          type: "array",
          items: { type: "string" },
          description: "List of specific fields or information that is missing and needed to proceed",
        },
        recommended_next_action: {
          type: "string",
          description: "One clear sentence describing what staff should do next",
        },
        draft_reply: {
          type: ["string", "null"],
          description: "The draft message body you sent via draft_message, or null if no message was drafted",
        },
        task_ids: {
          type: "array",
          items: { type: "string" },
          description: "IDs returned by create_task calls for this item",
        },
        escalation: {
          oneOf: [
            { type: "null" },
            {
              type: "object",
              properties: {
                reason: { type: "string" },
                severity: { type: "string", enum: ["P0", "P1"] },
              },
              required: ["reason", "severity"],
            },
          ],
        },
        decision_rationale: {
          type: "string",
          description: "2-3 sentence explanation of the key decisions made and why",
        },
      },
      required: [
        "item_id",
        "classification",
        "urgency",
        "requires_human_review",
        "extracted_intake",
        "missing_info",
        "recommended_next_action",
        "draft_reply",
        "task_ids",
        "escalation",
        "decision_rationale",
      ],
    },
  },
];

type SubmitInput = Omit<ItemOutput, "tools_called">;

const SAFEGUARDING_KEYWORDS = [
  "rough", "hit", "hitting", "hurt", "hurting", "abuse", "abused", "abusing",
  "neglect", "neglected", "harm", "harming", "unsafe", "danger", "scared",
  "afraid", "violence", "violent", "bruise", "bruised", "injure", "injured",
  "injury", "assault", "threatened", "threatening",
];

function hasSafeguardingSignal(item: InboxItem): boolean {
  const text = `${item.subject} ${item.body}`.toLowerCase();
  return SAFEGUARDING_KEYWORDS.some((kw) => text.includes(kw));
}

async function withRetry<T>(fn: () => Promise<T>, maxAttempts = 3): Promise<T> {
  let lastErr: unknown;
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
      if (attempt < maxAttempts - 1) {
        await new Promise((resolve) =>
          setTimeout(resolve, 1000 * Math.pow(2, attempt)),
        );
      }
    }
  }
  throw lastErr;
}

const CONCURRENCY = 3;

export async function runAgent(inbox: InboxItem[]): Promise<ItemOutput[]> {
  const results: (ItemOutput | null)[] = new Array(inbox.length).fill(null);
  const queue = inbox.map((item, index) => ({ item, index }));

  async function worker(): Promise<void> {
    while (true) {
      const next = queue.shift();
      if (!next) break;
      const { item, index } = next;
      process.stderr.write(`[triage] ${item.id}: ${item.subject}\n`);
      try {
        results[index] = await withItemContext(item.id, () => triageItem(item));
      } catch (err) {
        process.stderr.write(`[triage] ERROR ${item.id}: ${err instanceof Error ? err.message : String(err)}\n`);
        results[index] = fallbackOutput(item, err);
      }
    }
  }

  await Promise.all(Array.from({ length: CONCURRENCY }, worker));
  return results as ItemOutput[];
}

function fallbackOutput(item: InboxItem, err: unknown): ItemOutput {
  return {
    item_id: item.id,
    classification: "other",
    urgency: "P2",
    requires_human_review: true,
    extracted_intake: {
      child_name: null,
      dob_or_age: null,
      parent_contact: null,
      discipline: null,
      diagnosis_or_concern: null,
      payer: null,
      member_id: null,
    },
    missing_info: ["Agent error — manual triage required"],
    tools_called: [],
    recommended_next_action: "Manual triage required: agent failed to process this item.",
    draft_reply: null,
    task_ids: [],
    escalation: null,
    decision_rationale: `Agent error: ${err instanceof Error ? err.message : String(err)}`,
  };
}

async function triageItem(item: InboxItem): Promise<ItemOutput> {
  const safeguardingFlagged = hasSafeguardingSignal(item);
  const preamble = safeguardingFlagged
    ? "[SAFEGUARDING PRE-SCAN ALERT: This item contains language that may indicate harm or unsafe caregiving. Review carefully — classify as P0 if any doubt.]\n\n"
    : "";

  const messages: Anthropic.MessageParam[] = [
    {
      role: "user",
      content: `${preamble}Triage this inbox item. Use the appropriate tools, then call submit_triage with your final assessment.\n\n${JSON.stringify(item, null, 2)}`,
    },
  ];

  let submitted: SubmitInput | null = null;
  const MAX_ITERATIONS = 10;

  for (let i = 0; i < MAX_ITERATIONS && !submitted; i++) {
    const response = await withRetry(() =>
      client.messages.create({
        model: "claude-sonnet-4-6",
        max_tokens: 4096,
        system: [{ type: "text", text: SYSTEM_PROMPT, cache_control: { type: "ephemeral" } }],
        tools: TOOL_DEFINITIONS,
        messages,
      }),
    );

    const { input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens } = response.usage;
    process.stderr.write(
      `[triage] ${item.id} usage: in=${input_tokens} out=${output_tokens}` +
      (cache_read_input_tokens ? ` cache_read=${cache_read_input_tokens}` : "") +
      (cache_creation_input_tokens ? ` cache_write=${cache_creation_input_tokens}` : "") +
      "\n",
    );

    const toolResultContent: Anthropic.ToolResultBlockParam[] = [];

    for (const block of response.content) {
      if (block.type !== "tool_use") continue;

      if (block.name === "submit_triage") {
        submitted = block.input as SubmitInput;
        continue;
      }

      const result = await dispatchTool(
        block.name,
        block.input as Record<string, unknown>,
      );
      toolResultContent.push({
        type: "tool_result",
        tool_use_id: block.id,
        content: JSON.stringify(result),
      });
    }

    if (submitted) break;

    if (toolResultContent.length === 0) {
      throw new Error(
        `Agent stopped without submitting triage for ${item.id} (stop_reason: ${response.stop_reason})`,
      );
    }

    messages.push({ role: "assistant", content: response.content });
    messages.push({ role: "user", content: toolResultContent });
  }

  if (!submitted) {
    throw new Error(`Agent exceeded max iterations without submitting for ${item.id}`);
  }

  // Safety net: if the keyword pre-scan flagged this item but the main loop
  // under-classified it, force P0 so a safeguarding item can never ship as P2.
  if (safeguardingFlagged && submitted.urgency !== "P0") {
    process.stderr.write(
      `[triage] SAFEGUARDING OVERRIDE: ${item.id} upgraded from ${submitted.urgency} to P0\n`,
    );
    submitted = { ...submitted, urgency: "P0", requires_human_review: true };
  }

  const toolsCalled = getToolCallsForItem(item.id);

  // Derive task_ids and escalation from the audit trace rather than trusting
  // Claude's output fields, which could hallucinate IDs or omit entries.
  const taskIds = toolsCalled
    .filter((tc) => tc.name === "create_task")
    .map((tc) => tc.result_summary.match(/created task (\S+)/)?.[1])
    .filter((id): id is string => id !== undefined);

  const escalationCall = toolsCalled.find((tc) => tc.name === "escalate");
  const escalation = escalationCall
    ? {
        reason: escalationCall.args.reason as string,
        severity: escalationCall.args.severity as "P0" | "P1",
      }
    : null;

  return {
    ...submitted,
    task_ids: taskIds,
    escalation,
    tools_called: toolsCalled,
  };
}

async function dispatchTool(
  name: string,
  args: Record<string, unknown>,
): Promise<unknown> {
  switch (name) {
    case "search_patient":
      return search_patient(args as Parameters<typeof search_patient>[0]);
    case "verify_insurance":
      return verify_insurance(args as Parameters<typeof verify_insurance>[0]);
    case "lookup_policy":
      return lookup_policy(args as Parameters<typeof lookup_policy>[0]);
    case "find_slots":
      return find_slots(args as Parameters<typeof find_slots>[0]);
    case "hold_slot":
      return hold_slot(args as Parameters<typeof hold_slot>[0]);
    case "create_task":
      return create_task(args as Parameters<typeof create_task>[0]);
    case "draft_message":
      return draft_message(args as Parameters<typeof draft_message>[0]);
    case "escalate":
      return escalate(args as Parameters<typeof escalate>[0]);
    default:
      throw new Error(`Unknown tool: ${name}`);
  }
}
