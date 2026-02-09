import { Router, type Request, type Response, type IRouter } from "express";
import { trace, SpanStatusCode, context } from "@opentelemetry/api";

const router: IRouter = Router();
const tracer = trace.getTracer("demo-failures");

const SCENARIOS = [
  "timeout",
  "rate-limit",
  "context-overflow",
  "invalid-response",
  "hallucination",
  "auth-failure",
] as const;

type Scenario = (typeof SCENARIOS)[number];

interface FailureRequest {
  scenario?: Scenario;
  prompt?: string;
  model?: string;
}

const countTokens = (text: string): number => Math.ceil(text.length / 4);

router.post("/failures", async (req: Request, res: Response) => {
  const {
    prompt = "Tell me about quantum computing in extreme detail...",
    model = "gpt-4-mock",
  } = req.body as FailureRequest;

  // Pick scenario (random if not specified)
  let { scenario } = req.body as FailureRequest;
  if (!scenario || !SCENARIOS.includes(scenario)) {
    scenario = SCENARIOS[Math.floor(Math.random() * SCENARIOS.length)];
  }

  const parentSpan = tracer.startSpan("failures.simulate", undefined, context.active());

  try {
    parentSpan.setAttribute("failure.scenario", scenario);
    parentSpan.setAttribute("failure.type", "simulated");
    parentSpan.setAttribute("llm.model.name", model);
    parentSpan.setAttribute("llm.model.provider", "mock");

    // Child 1: Validate input (always succeeds)
    const validateSpan = tracer.startSpan(
      "failures.validate-input",
      undefined,
      trace.setSpan(context.active(), parentSpan)
    );
    const inputTokens = countTokens(prompt);
    validateSpan.setAttribute("input.token_count", inputTokens);
    validateSpan.setAttribute("input.prompt_length", prompt.length);
    await new Promise((r) => setTimeout(r, 10 + Math.random() * 20));
    validateSpan.setStatus({ code: SpanStatusCode.OK });
    validateSpan.end();

    // Child 2: LLM call (error happens here for most scenarios)
    const llmSpan = tracer.startSpan(
      "failures.llm-call",
      undefined,
      trace.setSpan(context.active(), parentSpan)
    );
    llmSpan.setAttribute("llm.model.name", model);
    llmSpan.setAttribute("gen_ai.usage.prompt_tokens", inputTokens);
    llmSpan.setAttribute("llm.usage.prompt_tokens", inputTokens);

    const result = await simulateScenario(scenario, llmSpan, model, inputTokens, prompt);

    // Child 3: Post-process (only for hallucination — the "success" path)
    if (result.success) {
      const postSpan = tracer.startSpan(
        "failures.post-process",
        undefined,
        trace.setSpan(context.active(), parentSpan)
      );
      postSpan.setAttribute("output.token_count", result.outputTokens ?? 0);
      if (result.warning) {
        postSpan.addEvent("hallucination_detected", {
          "hallucination.confidence_score": result.confidenceScore ?? 0,
          "hallucination.flagged": true,
        });
        postSpan.setAttribute("post_process.hallucination_flagged", true);
      }
      await new Promise((r) => setTimeout(r, 15 + Math.random() * 25));
      postSpan.setStatus({ code: SpanStatusCode.OK });
      postSpan.end();
    }

    // Set parent span status
    if (result.error) {
      parentSpan.setStatus({ code: SpanStatusCode.ERROR, message: result.error });
      parentSpan.recordException(new Error(result.error));
    } else {
      parentSpan.setStatus({ code: SpanStatusCode.OK });
    }

    const traceId = parentSpan.spanContext().traceId;
    const spanId = parentSpan.spanContext().spanId;

    // Return appropriate HTTP status
    const httpStatus = result.error ? (scenario === "auth-failure" ? 401 : scenario === "rate-limit" ? 429 : 500) : 200;

    res.status(httpStatus).json({
      success: !result.error,
      traceId,
      spanId,
      scenario,
      ...(result.error ? { error: result.error } : {}),
      ...(result.warning ? { warning: result.warning, confidenceScore: result.confidenceScore } : {}),
      ...(result.data ? { data: result.data } : {}),
    });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : "Unknown error";
    parentSpan.setStatus({ code: SpanStatusCode.ERROR, message: errorMessage });
    parentSpan.recordException(error as Error);
    res.status(500).json({
      success: false,
      traceId: parentSpan.spanContext().traceId,
      scenario,
      error: errorMessage,
    });
  } finally {
    parentSpan.end();
  }
});

// --- Scenario simulation ---

interface ScenarioResult {
  success: boolean;
  error?: string;
  warning?: string;
  confidenceScore?: number;
  outputTokens?: number;
  data?: Record<string, unknown>;
}

async function simulateScenario(
  scenario: Scenario,
  llmSpan: ReturnType<typeof tracer.startSpan>,
  model: string,
  inputTokens: number,
  prompt: string,
): Promise<ScenarioResult> {
  switch (scenario) {
    case "timeout":
      return simulateTimeout(llmSpan, model);
    case "rate-limit":
      return simulateRateLimit(llmSpan, model);
    case "context-overflow":
      return simulateContextOverflow(llmSpan, model, inputTokens, prompt);
    case "invalid-response":
      return simulateInvalidResponse(llmSpan, model);
    case "hallucination":
      return simulateHallucination(llmSpan, model, inputTokens);
    case "auth-failure":
      return simulateAuthFailure(llmSpan, model);
  }
}

async function simulateTimeout(llmSpan: any, model: string): Promise<ScenarioResult> {
  llmSpan.setAttribute("llm.request.timeout_ms", 3000);
  await new Promise((r) => setTimeout(r, 3000));
  const error = `Deadline exceeded: LLM response took >3000ms (model: ${model})`;
  llmSpan.setStatus({ code: SpanStatusCode.ERROR, message: error });
  llmSpan.recordException(new Error(error));
  llmSpan.end();
  return { success: false, error };
}

async function simulateRateLimit(llmSpan: any, model: string): Promise<ScenarioResult> {
  await new Promise((r) => setTimeout(r, 50 + Math.random() * 100));
  const error = `Rate limit exceeded: 429 Too Many Requests from provider (model: ${model})`;
  llmSpan.setAttribute("http.status_code", 429);
  llmSpan.setAttribute("llm.rate_limit.remaining", 0);
  llmSpan.setAttribute("llm.rate_limit.reset_seconds", 60);
  llmSpan.setStatus({ code: SpanStatusCode.ERROR, message: error });
  llmSpan.recordException(new Error(error));
  llmSpan.end();
  return { success: false, error };
}

async function simulateContextOverflow(
  llmSpan: any, model: string, inputTokens: number, prompt: string
): Promise<ScenarioResult> {
  await new Promise((r) => setTimeout(r, 30 + Math.random() * 50));
  const MAX_CONTEXT = 8192;
  const mockTokenCount = 32768; // Pretend the prompt is huge
  llmSpan.setAttribute("llm.context_window.max_tokens", MAX_CONTEXT);
  llmSpan.setAttribute("llm.context_window.requested_tokens", mockTokenCount);
  const error = `Context length exceeded: ${mockTokenCount} tokens > ${MAX_CONTEXT} max for ${model}`;
  llmSpan.setStatus({ code: SpanStatusCode.ERROR, message: error });
  llmSpan.recordException(new Error(error));
  llmSpan.end();
  return { success: false, error };
}

async function simulateInvalidResponse(llmSpan: any, model: string): Promise<ScenarioResult> {
  await new Promise((r) => setTimeout(r, 200 + Math.random() * 300));
  const garbled = "{\x00\x01invalid json \xFFresp";
  let error: string;
  try {
    JSON.parse(garbled);
    error = "Unexpected valid JSON"; // won't happen
  } catch (e) {
    error = `Invalid LLM response: ${(e as Error).message} (model: ${model})`;
  }
  llmSpan.setAttribute("llm.response.raw_length", garbled.length);
  llmSpan.setAttribute("llm.response.parse_error", true);
  llmSpan.setStatus({ code: SpanStatusCode.ERROR, message: error });
  llmSpan.recordException(new Error(error));
  llmSpan.end();
  return { success: false, error };
}

async function simulateHallucination(
  llmSpan: any, model: string, inputTokens: number
): Promise<ScenarioResult> {
  await new Promise((r) => setTimeout(r, 150 + Math.random() * 250));
  const outputTokens = 45 + Math.floor(Math.random() * 30);
  const confidenceScore = 0.15 + Math.random() * 0.2; // Low: 0.15-0.35
  llmSpan.setAttribute("gen_ai.usage.completion_tokens", outputTokens);
  llmSpan.setAttribute("llm.usage.completion_tokens", outputTokens);
  llmSpan.setAttribute("llm.confidence_score", confidenceScore);
  llmSpan.addEvent("hallucination_detected", {
    "hallucination.confidence_score": confidenceScore,
    "hallucination.threshold": 0.5,
    "hallucination.flagged": true,
  });
  // Span is OK (the call "succeeded") but flagged
  llmSpan.setStatus({ code: SpanStatusCode.OK });
  llmSpan.end();
  return {
    success: true,
    warning: "Low confidence response — possible hallucination",
    confidenceScore: Math.round(confidenceScore * 100) / 100,
    outputTokens,
    data: {
      response: "The quantum decoherence of neural network weights causes spontaneous token generation in the Hilbert space of transformer attention heads.",
      model,
    },
  };
}

async function simulateAuthFailure(llmSpan: any, model: string): Promise<ScenarioResult> {
  await new Promise((r) => setTimeout(r, 30 + Math.random() * 50));
  const error = `Authentication failed: 401 Unauthorized — invalid or expired API key (model: ${model})`;
  llmSpan.setAttribute("http.status_code", 401);
  llmSpan.setAttribute("llm.auth.method", "bearer_token");
  llmSpan.setStatus({ code: SpanStatusCode.ERROR, message: error });
  llmSpan.recordException(new Error(error));
  llmSpan.end();
  return { success: false, error };
}

export { router as failuresRouter };
