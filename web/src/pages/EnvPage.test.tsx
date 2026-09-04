// @vitest-environment jsdom
import { act, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { I18nProvider } from "@/i18n";
import type { EnvVarInfo } from "@/lib/api";

const apiMocks = vi.hoisted(() => ({
  deleteEnvVar: vi.fn(),
  getEnvVars: vi.fn(),
  revealEnvVar: vi.fn(),
  setEnvVar: vi.fn(),
}));
const setAfterTitle = vi.hoisted(() => vi.fn());

vi.mock("@/lib/api", () => ({ api: apiMocks }));
vi.mock("@/components/OAuthProvidersCard", () => ({
  OAuthProvidersCard: () => null,
}));
vi.mock("@/contexts/usePageHeader", () => ({
  usePageHeader: () => ({ setAfterTitle }),
}));
vi.mock("@/plugins", () => ({ PluginSlot: () => null }));

const skillCredential: EnvVarInfo = {
  advanced: true,
  category: "skill",
  description: "Credential used by the bundled Notion skill",
  is_password: true,
  is_set: true,
  redacted_value: "noti...1234",
  tools: [],
  url: null,
};

let container: HTMLDivElement;
let root: Root;

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT =
  true;

async function render(ui: ReactNode) {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(<I18nProvider>{ui}</I18nProvider>));
}

function buttonNamed(name: string): HTMLButtonElement {
  const button = Array.from(container.querySelectorAll("button")).find(
    (candidate) => candidate.textContent?.trim() === name,
  );
  if (!(button instanceof HTMLButtonElement)) {
    throw new Error(`Button not found: ${name}`);
  }
  return button;
}

beforeEach(() => {
  apiMocks.deleteEnvVar.mockReset();
  apiMocks.getEnvVars.mockReset();
  apiMocks.revealEnvVar.mockReset();
  apiMocks.setEnvVar.mockReset();
  setAfterTitle.mockReset();

  apiMocks.getEnvVars.mockResolvedValue({ NOTION_API_KEY: skillCredential });
  apiMocks.revealEnvVar.mockResolvedValue({
    key: "NOTION_API_KEY",
    value: "revealed-test-value",
  });
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
});

describe("EnvPage bundled-skill credentials", () => {
  it("renders them in a distinct section while preserving advanced filtering and reveal", async () => {
    const { default: EnvPage } = await import("./EnvPage");
    await render(<EnvPage />);

    await vi.waitFor(() =>
      expect(container.querySelector("#section-skill")).not.toBeNull(),
    );

    const skillSection = container.querySelector("#section-skill")!;
    expect(skillSection.textContent).toContain("Skills");
    expect(skillSection.textContent).toContain("NOTION_API_KEY");
    expect(skillSection.textContent).toContain("noti...1234");
    expect(skillSection.textContent).not.toContain("revealed-test-value");

    await act(async () => buttonNamed("Hide Advanced").click());
    expect(container.querySelector("#section-skill")).toBeNull();

    await act(async () => buttonNamed("Show Advanced").click());
    const revealButton = container.querySelector(
      '[aria-label="Reveal NOTION_API_KEY"]',
    );
    expect(revealButton).not.toBeNull();

    await act(async () => (revealButton as HTMLButtonElement).click());
    await vi.waitFor(() =>
      expect(apiMocks.revealEnvVar).toHaveBeenCalledWith("NOTION_API_KEY"),
    );
    expect(container.querySelector("#section-skill")?.textContent).toContain(
      "revealed-test-value",
    );

    const hideButton = container.querySelector(
      '[aria-label="Hide NOTION_API_KEY"]',
    );
    await act(async () => (hideButton as HTMLButtonElement).click());
    expect(container.querySelector("#section-skill")?.textContent).toContain(
      "noti...1234",
    );
  });
});
