import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import DOMPurify from "dompurify";
import latexCss from "../node_modules/latex.js/dist/css/article.css?inline";
import { useEffect, useRef, useState, type FormEvent } from "react";

type PaperMetadata = {
  title?: string | null;
  authors?: string | null;
  abstract?: string | null;
  categories?: string | null;
  primary_category?: string | null;
  doi?: string | null;
  journal_ref?: string | null;
  license?: string | null;
};

type SearchHit = {
  paper_id: string;
  metadata?: PaperMetadata | null;
  snippet: string;
  snippet_truncated: boolean;
};

type SearchResponse = {
  results: SearchHit[];
  search_url: string;
  metadata: {
    lookup_time_ms: number;
    result_count: number;
    offset: number;
    next_offset: number;
    has_more: boolean;
  };
};

type PaperText = {
  paper_id: string;
  metadata?: PaperMetadata | null;
  text: string;
  truncated: boolean;
  returned_chars: number;
  max_chars: number;
};

type EmbeddingStatus = "pending" | "embedding" | "done" | "failed";

type PaperChunk = {
  chunk_id: string;
  text: string;
  byte_range: [number, number];
  word_count: number;
  cost: number;
};

type ChunkItem = PaperChunk & {
  embeddingStatus: EmbeddingStatus;
  embeddingError?: string;
};

type ChunkResponse = {
  paper_id: string;
  chunk_table: string;
  token_budget: number;
  source_chars: number;
  source_truncated: boolean;
  chunks: ChunkItem[];
};

type ChunkApiResponse = Omit<ChunkResponse, "chunks"> & {
  chunks: PaperChunk[];
};

type IndexJobProgress = {
  job_id: string;
  state: "queued" | "running" | "paused" | "complete" | "complete_with_errors" | "failed";
  total_papers: number;
  total_papers_known: boolean;
  phase_completed: number;
  phase_total: number | null;
  scanned_papers: number;
  elapsed_seconds: number;
  papers_per_second: number;
  eta_seconds: number | null;
  chunked_papers: number;
  embedded_papers: number;
  failed_papers: number;
  current_paper: string | null;
  phase: string | null;
  error: string | null;
};

type StartIndexJobResponse = {
  job_id: string;
  events_url: string;
  state: string;
};

type AbstractEmbeddingIndexStatus = {
  embedding_model: string;
  vector_column: string;
  vector_column_exists: boolean;
  readiness_column: string;
  readiness_index_ready: boolean;
  has_embeddings: boolean;
  embedded_vectors: number;
  indexed_vectors: number;
  unindexed_vectors: number;
  index_unindexed_rows: number;
  vector_index_ready: boolean;
};

type IndexJobSubscription = {
  job: StartIndexJobResponse;
  paperId?: string;
  mode: EmbeddingMode;
  model: string;
};

type AvailableModel = {
  id: string;
  tags: string[];
};

type ModelChoices = {
  embedding: AvailableModel[];
  reranking: AvailableModel[];
};

type EmbeddingMode = "full_text" | "abstract";

type QueryPipelineResponse = {
  embedding_model: string;
  reranker_model: string | null;
  hits: Array<{
    chunk_id: string;
    paper_id: string;
    text: string;
    lexical_rank: number | null;
    semantic_rank: number | null;
    fused_score: number;
    rerank_score: number | null;
  }>;
};

function getText(result: { content: Array<{ type: string; text?: string }>; isError?: boolean }) {
  if (result.isError) {
    throw new Error(result.content.map((item) => item.text ?? "").join("\n") || "MCP tool failed");
  }
  return result.content
    .filter((item) => item.type === "text")
    .map((item) => item.text ?? "")
    .join("\n");
}

function HighlightText({ text, query }: { text: string; query: string }) {
  const terms = query.trim().split(/\s+/).filter(Boolean).sort((a, b) => b.length - a.length);
  if (!terms.length) return text;

  const pattern = new RegExp(`(${terms.map(escapeRegExp).join("|")})`, "giu");
  return text.split(pattern).map((part, index) =>
    terms.some((term) => part.toLocaleLowerCase() === term.toLocaleLowerCase()) ? (
      <mark className="query-match" key={`${part}-${index}`}>{part}</mark>
    ) : (
      part
    ),
  );
}

function escapeRegExp(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function formatDuration(seconds: number | null) {
  if (seconds === null || !Number.isFinite(seconds)) return "calculating…";
  const totalSeconds = Math.max(0, Math.ceil(seconds));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const remainder = totalSeconds % 60;
  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return `${minutes}m ${remainder}s`;
  return `${remainder}s`;
}

function LatexSnippet({ source, query, renderLatex }: { source: string; query: string; renderLatex: boolean }) {
  const [html, setHtml] = useState("");

  useEffect(() => {
    if (!renderLatex) {
      setHtml("");
      return;
    }
    let cancelled = false;
    void import("latex.js")
      .then(({ HtmlGenerator, parse }) => {
        let body: string | undefined;
        for (const candidate of snippetDocuments(source)) {
          try {
            const document = renderWithLatexFallback(candidate, (normalized) =>
              parse(normalized, {
                generator: new HtmlGenerator({ hyphenate: false }),
              }).htmlDocument(),
            );
            body = document.body.innerHTML;
            break;
          } catch {
            continue;
          }
        }
        if (!body) throw new Error("Could not render this excerpt with LaTeX.js");
        body = highlightHtml(body, query);
        const safeBody = DOMPurify.sanitize(body, {
          FORBID_TAGS: ["script", "iframe", "object", "embed", "form"],
        });
        if (!cancelled) setHtml(snippetDocument(safeBody));
      })
      .catch(() => {
        if (!cancelled) setHtml("");
      });
    return () => {
      cancelled = true;
    };
  }, [source, query, renderLatex]);

  if (!renderLatex) {
    return <span className="snippet-source"><HighlightText text={source.replace(/\s+/g, " ").trim()} query={query} /></span>;
  }
  if (!html) {
    return <span className="snippet-source"><HighlightText text={source.replace(/\s+/g, " ").trim()} query={query} /></span>;
  }
  return <iframe className="snippet-frame" title="Rendered search excerpt" sandbox="allow-same-origin" srcDoc={html} />;
}

function PaperRenderer({ source, renderLatex }: { source: string; renderLatex: boolean }) {
  const [rendered, setRendered] = useState<{ html: string; error: string }>({
    html: "",
    error: "",
  });

  useEffect(() => {
    if (!renderLatex) return;
    let cancelled = false;
    setRendered({ html: "", error: "" });
    void import("latex.js")
      .then(({ HtmlGenerator, parse }) => {
        const document = renderWithLatexFallback(source, (normalized) =>
          parse(normalized, {
            generator: new HtmlGenerator({ hyphenate: false }),
          }).htmlDocument(),
        );
        const body = DOMPurify.sanitize(document.body.innerHTML, {
          FORBID_TAGS: ["script", "iframe", "object", "embed", "form"],
        });
        if (!cancelled) {
          setRendered({
            html: paperDocument(body),
            error: "",
          });
        }
      })
      .catch((reason: unknown) => {
        if (!cancelled) {
          setRendered({
            html: "",
            error: reason instanceof Error ? reason.message : String(reason),
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [source, renderLatex]);

  if (!renderLatex) {
    return <pre className="paper-source">{source}</pre>;
  }

  if (rendered.error) {
    return (
      <div className="render-fallback">
        <p>Could not render this source as a complete LaTeX document. Showing its source instead.</p>
        <details>
          <summary>LaTeX.js parser error</summary>
          <pre>{rendered.error}</pre>
        </details>
        <pre className="paper-source">{source}</pre>
      </div>
    );
  }

  if (!rendered.html) {
    return <div className="render-loading">Preparing the paper layout…</div>;
  }

  return (
    <iframe
      className="latex-frame"
      title="Rendered paper"
      sandbox="allow-same-origin"
      srcDoc={rendered.html}
    />
  );
}

function toArticleDocument(source: string) {
  const documentStart = source.indexOf("\\begin{document}");
  const documentEnd = source.lastIndexOf("\\end{document}");
  const bodyStart = documentStart + "\\begin{document}".length;
  let body = documentStart < 0
    ? source
    : source.slice(bodyStart, documentEnd < bodyStart ? undefined : documentEnd);
  const operators = [...source.matchAll(/\\DeclareMathOperator\*?\s*\{\\([A-Za-z]+)\}\s*\{([^{}]*)\}/g)];
  for (const [, name, value] of operators) {
    body = body.replace(new RegExp(`\\\\${escapeRegExp(name)}(?![A-Za-z])`, "g"), `\\mathrm{${value}}`);
  }
  body = body
    .replace(/\\(?:cite[a-z]*|parencite|textcite|autocite)(?:\[[^\]]*\])*\s*\{([^{}]*)\}/gi, " [$1]")
    .replace(/\\eqref\{([^{}]*)\}/g, "($1)")
    .replace(/\\(?:ref|autoref|cref|Cref)\{([^{}]*)\}/g, "[$1]")
    .replace(/\\(?:label|bibliography|bibliographystyle|addbibresource|nocite)\{[^{}]*\}/g, "")
    .replace(/\\url\{([^{}]*)\}/g, "\\texttt{$1}")
    .replace(/\\href\{[^{}]*\}\{([^{}]*)\}/g, "$1")
    .replace(/\\includegraphics\*?(?:\[[^\]]*\])?\s*\{[^{}]*\}/g, "")
    .replace(/\\begin\{thebibliography\}(?:\s*\{[^{}]*\})?[\s\S]*?\\end\{thebibliography\}/g, "")
    .replace(/\\usepackage(?:\[[^\]]*\])?\{[^}]*\}/g, "");
  return `\\documentclass{article}\n\\begin{document}\n${body}\n\\end{document}`;
}

function renderWithLatexFallback(source: string, parseDocument: (normalized: string) => Document) {
  let normalized = toArticleDocument(source);
  let lastError: unknown;
  for (let attempt = 0; attempt < 64; attempt++) {
    try {
      return parseDocument(normalized);
    } catch (reason) {
      lastError = reason;
      const message = reason instanceof Error ? reason.message : String(reason);
      const unknownMacro = /unknown macro: \\([A-Za-z]+)\b/.exec(message)?.[1];
      const unknownEnvironment = /unknown environment: ([A-Za-z]+\*?)/.exec(message)?.[1];
      let next = normalized;
      if (unknownMacro) {
        const pattern = new RegExp(`\\\\${escapeRegExp(unknownMacro)}(?![A-Za-z])`, "g");
        next = normalized.replace(pattern, "");
      } else if (unknownEnvironment) {
        const env = escapeRegExp(unknownEnvironment);
        const beginPattern = new RegExp(`\\\\begin\\{${env}\\}(?:\\s*\\[[^\\]]*\\])?(?:\\s*\\{[^{}]*\\})?`, "g");
        const endPattern = new RegExp(`\\\\end\\{${env}\\}`, "g");
        next = normalized.replace(beginPattern, "").replace(endPattern, "");
      } else if (/\\end\{document\} missing/.test(message)) {
        next = normalized
          .replace(/\\begin\{(?!document\})[^}]+\}(?:\s*\[[^\]]*\])?(?:\s*\{[^{}]*\})?/g, "")
          .replace(/\\end\{(?!document\})[^}]+\}/g, "");
      } else {
        throw reason;
      }
      if (next === normalized) throw reason;
      normalized = next;
    }
  }
  throw lastError ?? new Error("LaTeX.js could not parse the document");
}

function snippetDocuments(source: string) {
  const normalized = source
    .replace(/^\s*…/, "")
    .replace(/…\s*$/, "")
    .replace(/\\documentclass(?:\[[^\]]*\])?\{[^}]*\}/g, "")
    .replace(/\\usepackage(?:\[[^\]]*\])?\{[^}]*\}/g, "")
    .replace(/\\begin\{document\}/g, "")
    .replace(/\\end\{document\}/g, "");
  return [
    source,
    `\\documentclass{article}\\begin{document}${normalized}\\end{document}`,
  ];
}

function highlightHtml(html: string, query: string) {
  const terms = query.trim().split(/\s+/).filter(Boolean).sort((a, b) => b.length - a.length);
  if (!terms.length) return html;

  const expression = new RegExp(terms.map(escapeRegExp).join("|"), "giu");
  const parsed = new DOMParser().parseFromString(html, "text/html");
  const walker = parsed.createTreeWalker(parsed.body, NodeFilter.SHOW_TEXT);
  const textNodes: Text[] = [];
  while (walker.nextNode()) {
    if (!walker.currentNode.parentElement?.closest("script,style,mark")) {
      textNodes.push(walker.currentNode as Text);
    }
  }

  for (const node of textNodes) {
    const value = node.nodeValue ?? "";
    expression.lastIndex = 0;
    let match: RegExpExecArray | null;
    let offset = 0;
    const fragment = parsed.createDocumentFragment();
    while ((match = expression.exec(value)) !== null) {
      fragment.append(value.slice(offset, match.index));
      const mark = parsed.createElement("mark");
      mark.textContent = match[0];
      fragment.append(mark);
      offset = match.index + match[0].length;
    }
    if (offset > 0) {
      fragment.append(value.slice(offset));
      node.replaceWith(fragment);
    }
  }
  return parsed.body.innerHTML;
}

function paperDocument(body: string) {
  return `<!doctype html><html><head><meta charset="utf-8"><style>${latexCss}${paperCss}</style></head><body>${body}</body></html>`;
}

function snippetDocument(body: string) {
  const compactStyle = `
    html, body { overflow-x: hidden; scrollbar-width: none; }
    *::-webkit-scrollbar { display: none; }
    body { max-width: none; margin: 0; padding: 3px 5px; font: 12px/1.4 Georgia, serif; overflow-wrap: anywhere; word-break: break-word; }
    p { margin: 0; text-align: left; }
    h1, h2, h3, h4 { margin: 0; font-size: 1em; }
    mark { background: #ffdf70; color: #111; }
  `;
  return `<!doctype html><html><head><meta charset="utf-8"><style>${latexCss}${compactStyle}</style></head><body>${body}</body></html>`;
}

const paperCss = `
  html { background: #fff; color: #111; font: 16px/1.55 Georgia, "Times New Roman", serif; overflow-x: hidden; scrollbar-width: none; }
  body { max-width: 1100px; margin: 0 auto; padding: 20px 30px 48px; overflow-x: hidden; overflow-wrap: anywhere; word-break: break-word; }
  *::-webkit-scrollbar { display: none; }
  h1, h2, h3, h4 { line-height: 1.25; margin: 1.25em 0 .45em; }
  h1 { font-size: 1.65em; }
  h2 { font-size: 1.35em; border-bottom: 1px solid #ddd; padding-bottom: .2em; }
  h3 { font-size: 1.15em; }
  p { margin: .65em 0; }
  table { max-width: 100%; border-collapse: collapse; }
  img, svg { max-width: 100%; height: auto; }
  pre { white-space: pre-wrap; overflow-wrap: anywhere; word-break: break-word; }
  @media (max-width: 700px) { body { padding: 16px 12px 36px; } }
`;

function App({ apiBaseUrl }: { apiBaseUrl: string }) {
  const initialParams = new URLSearchParams(window.location.search);
  const serverAddress = new URL(apiBaseUrl).host;
  const clientRef = useRef<Client | null>(null);
  const transportRef = useRef<StreamableHTTPClientTransport | null>(null);
  const jobEventSourceRef = useRef<EventSource | null>(null);
  const indexJobSubscriptionRef = useRef<IndexJobSubscription | null>(null);
  const lastIndexProgressAtRef = useRef(0);
  const lastIndexProgressRenderAtRef = useRef(0);
  const [query, setQuery] = useState(initialParams.get("query") ?? "");
  const [lexicalSearch, setLexicalSearch] = useState(
    initialParams.get("lexical") === null || initialParams.get("lexical") === "1",
  );
  const [semanticSearch, setSemanticSearch] = useState(
    initialParams.get("semantic") === null
      ? initialParams.get("embedded_only") === "1"
      : initialParams.get("semantic") === "1",
  );
  const [searchUrl, setSearchUrl] = useState("");
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [lookupTimeMs, setLookupTimeMs] = useState<number | null>(null);
  const [nextOffset, setNextOffset] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  const [selected, setSelected] = useState<PaperText | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState(`Connecting to ${serverAddress}…`);
  const [renderLatex, setRenderLatex] = useState(false);
  const [chunkView, setChunkView] = useState(false);
  const [chunkResult, setChunkResult] = useState<ChunkResponse | null>(null);
  const [chunkBusy, setChunkBusy] = useState(false);
  const [embeddingBusy, setEmbeddingBusy] = useState(false);
  const [indexJobControlBusy, setIndexJobControlBusy] = useState(false);
  const [chunkError, setChunkError] = useState("");
  const [embeddingModel, setEmbeddingModel] = useState("");
  const [indexJobProgress, setIndexJobProgress] = useState<IndexJobProgress | null>(null);
  const [indexJobMessage, setIndexJobMessage] = useState("");
  const [rerankQuery, setRerankQuery] = useState("");
  const [queryPipelineBusy, setQueryPipelineBusy] = useState(false);
  const [queryPipelineResult, setQueryPipelineResult] = useState<QueryPipelineResponse | null>(null);
  const [modelChoices, setModelChoices] = useState<ModelChoices>({ embedding: [], reranking: [] });
  const [modelChoicesError, setModelChoicesError] = useState("");
  const [selectedEmbeddingModel, setSelectedEmbeddingModel] = useState("");
  const [abstractEmbeddingIndexStatus, setAbstractEmbeddingIndexStatus] =
    useState<AbstractEmbeddingIndexStatus | null>(null);
  const [abstractIndexBusy, setAbstractIndexBusy] = useState(false);
  const [abstractIndexError, setAbstractIndexError] = useState("");
  const [selectedRerankerModel, setSelectedRerankerModel] = useState("");
  const [embeddingMode, setEmbeddingMode] = useState<EmbeddingMode>("abstract");
  const [tokenBudget, setTokenBudget] = useState(480);

  useEffect(() => {
    if (!selectedEmbeddingModel) return;
    void refreshAbstractEmbeddingIndexStatus(selectedEmbeddingModel);
  }, [selectedEmbeddingModel]);

  useEffect(() => {
    if (!embeddingBusy) return;
    const timer = window.setInterval(() => {
      const events = jobEventSourceRef.current;
      if (events && events.readyState !== EventSource.CLOSED &&
        Date.now() - lastIndexProgressAtRef.current >= 15_000) {
        setIndexJobMessage("No progress update received for 15 seconds. The backend may still be working; reconnect to check.");
      }
    }, 1000);
    return () => window.clearInterval(timer);
  }, [embeddingBusy]);

  useEffect(() => {
    let active = true;
    const client = new Client({ name: "arxiv-search-ui", version: "0.1.0" });
    const transport = new StreamableHTTPClientTransport(
      new URL(apiBaseUrl),
    );
    clientRef.current = client;
    transportRef.current = transport;

    void client
      .connect(transport)
      .then(async () => {
        if (!active) return;
        setStatus(`Connected to ${serverAddress}`);
        try {
          const result = await client.callTool({ name: "dataset_status", arguments: {} });
          if (!active) return;
          const info = JSON.parse(getText(result as { content: Array<{ type: string; text?: string }>; isError?: boolean })) as {
            text_rows: number;
            metadata_rows: number;
          };
          setStatus(`Connected to ${serverAddress} · ${info.text_rows.toLocaleString()} papers · ${info.metadata_rows.toLocaleString()} metadata records`);
          const activeJobResponse = await fetch(new URL("/indexing-jobs/current", apiBaseUrl));
          if (activeJobResponse.ok) {
            const activeJob = await activeJobResponse.json() as IndexJobProgress | null;
            if (active && activeJob) {
              setIndexJobProgress(activeJob);
              setEmbeddingBusy(activeJob.state === "running" || activeJob.state === "queued");
              connectIndexingProgress({
                job: {
                  job_id: activeJob.job_id,
                  events_url: `/indexing-jobs/${activeJob.job_id}/events`,
                  state: activeJob.state,
                },
                mode: embeddingMode,
                model: selectedEmbeddingModel,
              });
                setEmbeddingBusy(activeJob.state === "running" || activeJob.state === "queued");
            }
          }
          try {
            const modelsResult = await client.callTool({ name: "list_models", arguments: {} });
            const choices = JSON.parse(
              getText(modelsResult as { content: Array<{ type: string; text?: string }>; isError?: boolean }),
            ) as ModelChoices;
            setModelChoices(choices);
            setSelectedEmbeddingModel((current) =>
              choices.embedding.some((model) => model.id === current)
                ? current
                : choices.embedding.find((model) => model.id === "Qwen3-Embedding-8B-Q4_KM-32K")?.id
                  ?? choices.embedding[0]?.id
                  ?? "",
            );
            setSelectedRerankerModel((current) =>
              choices.reranking.some((model) => model.id === current)
                ? current
                : choices.reranking.find((model) => model.id === "Qwen3-Reranker-8B-Q4_K_M-32K")?.id
                  ?? choices.reranking[0]?.id
                  ?? "",
            );
          } catch (reason) {
            setModelChoicesError(
              `Could not load model list: ${reason instanceof Error ? reason.message : String(reason)}`,
            );
          }
          const tool = initialParams.get("tool");
          if ((tool === "search_full_text" && initialParams.get("query")) ||
            (tool === "search_embedded_papers" &&
              (initialParams.get("embedded_only") === "1" || initialParams.get("semantic") === "1"))) {
            await fetchSearchPage(
              Number.parseInt(initialParams.get("offset") ?? "0", 10) || 0,
              false,
              Number.parseInt(initialParams.get("limit") ?? "10", 10) || 10,
              false,
            );
          } else if (tool === "get_paper_text" && initialParams.get("paper_id")) {
            await openPaper(
              initialParams.get("paper_id")!,
              Number.parseInt(initialParams.get("max_chars") ?? "1000000", 10) || 1_000_000,
              false,
            );
          }
        } catch (reason) {
          if (!active) return;
          setError(reason instanceof Error ? reason.message : String(reason));
        }
      })
      .catch((reason: unknown) => {
        if (!active) return;
        setError(reason instanceof Error ? reason.message : String(reason));
        setStatus(`Not connected to ${serverAddress}`);
      });

    return () => {
      active = false;
      jobEventSourceRef.current?.close();
      jobEventSourceRef.current = null;
      void client.close();
      clientRef.current = null;
      transportRef.current = null;
    };
  }, [apiBaseUrl, serverAddress]);

  async function fetchSearchPage(offset: number, append: boolean, limit = 10, updateUrl = true) {
    const client = clientRef.current;
    const searchQuery = query.trim();
    const searchMode = lexicalSearch
      ? semanticSearch ? "hybrid" : "lexical"
      : semanticSearch ? "semantic" : null;
    if (!client || !searchMode || (!searchQuery && !semanticSearch)) return;

    setBusy(true);
    setError("");
    setSelected(null);
    try {
      const response = await client.callTool({
        name: searchMode === "lexical" ? "search_full_text" : "search_embedded_papers",
        arguments: searchMode !== "lexical"
          ? {
              query: searchQuery,
              ...(selectedEmbeddingModel ? { embedding_model: selectedEmbeddingModel } : {}),
              ...(searchMode === "semantic" ? { semantic_only: true } : {}),
              limit,
              offset,
            }
          : { query: searchQuery, limit, offset },
      }, undefined, { timeout: 180_000 });
      const searchResponse = JSON.parse(
        getText(response as { content: Array<{ type: string; text?: string }>; isError?: boolean }),
      ) as SearchResponse;
      setHits((current) => append ? [...current, ...searchResponse.results] : searchResponse.results);
      setLookupTimeMs(searchResponse.metadata.lookup_time_ms);
      setNextOffset(searchResponse.metadata.next_offset);
      setHasMore(searchResponse.metadata.has_more);
      const shareUrl = new URL(searchResponse.search_url, window.location.origin);
      shareUrl.searchParams.set("lexical", lexicalSearch ? "1" : "0");
      shareUrl.searchParams.set("semantic", semanticSearch ? "1" : "0");
      setSearchUrl(shareUrl.toString());
      if (updateUrl) {
        window.history.pushState(null, "", `${shareUrl.pathname}${shareUrl.search}`);
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
      if (!append) {
        setHits([]);
        setLookupTimeMs(null);
        setNextOffset(0);
        setHasMore(false);
      }
    } finally {
      setBusy(false);
    }
  }

  async function refreshAbstractEmbeddingIndexStatus(model = selectedEmbeddingModel) {
    const client = clientRef.current;
    if (!client || !model) return;
    try {
      const response = await client.callTool({
        name: "abstract_embedding_index_status",
        arguments: { embedding_model: model },
      });
      const indexStatus = JSON.parse(
        getText(response as { content: Array<{ type: string; text?: string }>; isError?: boolean }),
      ) as AbstractEmbeddingIndexStatus;
      if (indexStatus.embedding_model !== model) return;
      setAbstractEmbeddingIndexStatus(indexStatus);
      setAbstractIndexError("");
    } catch (reason) {
      setAbstractEmbeddingIndexStatus(null);
      setAbstractIndexError(
        reason instanceof Error ? reason.message : String(reason),
      );
    }
  }

  async function buildAbstractVectorIndex() {
    const client = clientRef.current;
    if (!client || !selectedEmbeddingModel || abstractIndexBusy || embeddingBusy) return;
    setAbstractIndexBusy(true);
    setAbstractIndexError("");
    try {
      const response = await client.callTool({
        name: "build_abstract_vector_index",
        arguments: { embedding_model: selectedEmbeddingModel },
      }, undefined, { timeout: 3_600_000 });
      const builtStatus = JSON.parse(
        getText(response as { content: Array<{ type: string; text?: string }>; isError?: boolean }),
      ) as AbstractEmbeddingIndexStatus;
      setAbstractEmbeddingIndexStatus(builtStatus);
    } catch (reason) {
      setAbstractIndexError(
        reason instanceof Error ? reason.message : String(reason),
      );
      await refreshAbstractEmbeddingIndexStatus();
    } finally {
      setAbstractIndexBusy(false);
    }
  }

  async function search(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSelected(null);
    await fetchSearchPage(0, false);
  }

  async function loadMore() {
    await fetchSearchPage(nextOffset, true);
  }

  async function openPaper(paperId: string, maxChars = 1_000_000, updateUrl = true) {
    const client = clientRef.current;
    if (!client) return;
    setBusy(true);
    setError("");
    setChunkView(false);
    setChunkResult(null);
    setChunkError("");
    try {
      const response = await client.callTool({
        name: "get_paper_text",
        arguments: { paper_id: paperId, max_chars: maxChars },
      });
      setSelected(
        JSON.parse(getText(response as { content: Array<{ type: string; text?: string }>; isError?: boolean })) as PaperText,
      );
      if (updateUrl) {
        const params = new URLSearchParams({
          tool: "get_paper_text",
          paper_id: paperId,
          max_chars: String(maxChars),
        });
        window.history.pushState(null, "", `/?${params.toString()}`);
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  }

  function closePaper() {
    setSelected(null);
    setChunkView(false);
    setChunkResult(null);
    const returnUrl = searchUrl
      ? new URL(searchUrl)
      : new URL("/", window.location.origin);
    window.history.pushState(
      null,
      "",
      `${returnUrl.pathname}${returnUrl.search}${returnUrl.hash}`,
    );
  }

  async function chunkSelectedPaper() {
    const client = clientRef.current;
    if (!client || !selected) return;
    setChunkBusy(true);
    setChunkError("");
    try {
      const response = await client.callTool({
        name: "chunk_paper",
        arguments: {
          paper_id: selected.paper_id,
          token_budget: tokenBudget,
        },
      });
      const result = JSON.parse(
        getText(response as { content: Array<{ type: string; text?: string }>; isError?: boolean }),
      ) as ChunkApiResponse;
      setChunkResult({
        ...result,
        chunks: result.chunks.map((chunk) => ({
          ...chunk,
          embeddingStatus: "pending",
        })),
      });
      setEmbeddingModel("");
      setChunkView(true);
    } catch (reason) {
      setChunkError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setChunkBusy(false);
    }
  }

  async function startIndexingJob(paperId?: string) {
    const client = clientRef.current;
    if (!client) return;
    const activeJob = indexJobProgress;
    if (
      activeJob &&
      ["queued", "running", "paused"].includes(activeJob.state)
    ) {
      if (activeJob.state === "paused") {
        await setIndexingPaused(false, activeJob);
      }
      return;
    }
    if (embeddingBusy) return;
    jobEventSourceRef.current?.close();
    jobEventSourceRef.current = null;
    setEmbeddingBusy(true);
    setChunkError("");
    setIndexJobProgress(null);
    indexJobSubscriptionRef.current = null;
    setIndexJobMessage("Submitting indexing job…");
    try {
      const response = await client.callTool({
        name: "start_indexing_job",
        arguments: paperId
          ? { paper_id: paperId, token_budget: tokenBudget, mode: embeddingMode }
          : { mode: embeddingMode },
      });
      const job = JSON.parse(
        getText(response as { content: Array<{ type: string; text?: string }>; isError?: boolean }),
      ) as StartIndexJobResponse;
      connectIndexingProgress({ job, paperId, mode: embeddingMode, model: selectedEmbeddingModel });
    } catch (reason) {
      setEmbeddingBusy(false);
      const message = reason instanceof Error ? reason.message : String(reason);
      setChunkError(message);
      setIndexJobMessage(`Could not start indexing: ${message}`);
    }
  }

  function connectIndexingProgress(subscription: IndexJobSubscription) {
      jobEventSourceRef.current?.close();
      indexJobSubscriptionRef.current = subscription;
      const { job, paperId, mode, model } = subscription;
      setEmbeddingBusy(true);
      setIndexJobMessage("Connecting to indexing progress…");
      lastIndexProgressAtRef.current = Date.now();
      const events = new EventSource(new URL(job.events_url, apiBaseUrl));
      jobEventSourceRef.current = events;
      events.addEventListener("progress", (event) => {
        let progress: IndexJobProgress;
        try {
          progress = JSON.parse((event as MessageEvent<string>).data) as IndexJobProgress;
        } catch (reason) {
          events.close();
          jobEventSourceRef.current = null;
          setEmbeddingBusy(false);
          setIndexJobMessage(`Invalid indexing progress: ${reason instanceof Error ? reason.message : String(reason)}. The backend job may still be running.`);
          return;
        }
        lastIndexProgressAtRef.current = Date.now();
        setIndexJobMessage("");
        const terminal = progress.state === "complete" ||
          progress.state === "complete_with_errors" ||
          progress.state === "failed";
        const now = Date.now();
        if (terminal || now - lastIndexProgressRenderAtRef.current >= 2000) {
          lastIndexProgressRenderAtRef.current = now;
          setIndexJobProgress(progress);
        }
        if (terminal) {
          events.close();
          if (jobEventSourceRef.current === events) jobEventSourceRef.current = null;
          indexJobSubscriptionRef.current = null;
          setEmbeddingBusy(false);
          if (mode === "abstract") {
            void refreshAbstractEmbeddingIndexStatus(model);
          }
          if (
            paperId &&
            mode === "full_text" &&
            progress.embedded_papers > 0 &&
            progress.failed_papers === 0
          ) {
            setEmbeddingModel(model);
            setChunkResult((current) => current && ({
              ...current,
              chunks: current.chunks.map((chunk) => ({
                ...chunk,
                embeddingStatus: "done",
                embeddingError: undefined,
              })),
            }));
          }
          if (progress.failed_papers > 0 || progress.error) {
            setChunkError(
              progress.error ??
                `${progress.failed_papers} paper${progress.failed_papers === 1 ? "" : "s"} had embedding failures. Retry the job to resume missing vectors.`,
            );
          }
        }
      });
      events.onopen = () => {
        lastIndexProgressAtRef.current = Date.now();
        setIndexJobMessage("Connected; waiting for indexing progress…");
      };
      events.onerror = () => {
        if (events.readyState === EventSource.CLOSED) {
          setEmbeddingBusy(false);
          setIndexJobMessage("Indexing progress disconnected. The backend job may still be running.");
        } else {
          setIndexJobMessage("Reconnecting to indexing progress… The backend job continues running.");
        }
      };
  }

  async function setIndexingPaused(paused: boolean, job = indexJobProgress) {
    const client = clientRef.current;
    if (!client || !job || indexJobControlBusy) return;
    setIndexJobControlBusy(true);
    setChunkError("");
    try {
      const response = await client.callTool({
        name: "set_indexing_job_paused",
        arguments: { job_id: job.job_id, paused },
      });
      getText(response as { content: Array<{ type: string; text?: string }>; isError?: boolean });
      setIndexJobProgress((current) =>
        current?.job_id === job.job_id
          ? { ...current, state: paused ? "paused" : "running" }
          : current,
      );
      if (!paused) {
        setIndexJobMessage("");
      }

    } catch (reason) {
      setChunkError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setIndexJobControlBusy(false);
    }
  }

  async function indexOrResume(paperId?: string) {
    if (indexJobProgress?.state === "paused") {
      await setIndexingPaused(false, indexJobProgress);
      return;
    }
    if (indexJobProgress?.state === "running") {
      await setIndexingPaused(true, indexJobProgress);
      return;
    }
    if (indexJobProgress?.state === "queued") return;
    await startIndexingJob(paperId);
  }

  async function embedPaperChunks() {
    if (!selected || !chunkResult || chunkResult.chunks.length === 0) return;
    await startIndexingJob(selected.paper_id);
  }

  async function indexSelectedPaper() {
    if (!selected) return;
    if (indexJobProgress && ["running", "paused", "queued"].includes(indexJobProgress.state)) {
      await indexOrResume(selected.paper_id);
      return;
    }
    if (embeddingMode === "full_text") {
      await embedPaperChunks();
    } else {
      await startIndexingJob(selected.paper_id);
    }
  }

  async function startFullIndexingJob() {
    const target = embeddingMode === "abstract"
      ? "Queue every paper whose abstract has no vector?"
      : "Queue every paper that is not fully chunked and embedded?";
    if (window.confirm(`${target} This may run for a long time.`)) {
      await startIndexingJob();
    }
  }

  async function testQueryPipeline() {
    const client = clientRef.current;
    if (!client || !selected || !rerankQuery.trim() || queryPipelineBusy) return;
    if (embeddingMode === "full_text" && !chunkResult) return;

    setQueryPipelineBusy(true);
    setChunkError("");
    setQueryPipelineResult(null);
    try {
      const response = await client.callTool({
        name: "search_hybrid_chunks",
        arguments: {
          chunk_table: chunkResult?.chunk_table ?? "arxiv_chunks",
          query: rerankQuery.trim(),
          embedding_model: selectedEmbeddingModel,
          mode: embeddingMode,
          reranker_model: selectedRerankerModel || undefined,
          candidate_limit: 30,
        },
      });
      setQueryPipelineResult(JSON.parse(
        getText(response as { content: Array<{ type: string; text?: string }>; isError?: boolean }),
      ) as QueryPipelineResponse);
    } catch (reason) {
      setChunkError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setQueryPipelineBusy(false);
    }
  }

  return (
    <main className="page">
      <header className="topbar">
        <a className="brand" href="/" aria-label="ArXiv search home">
          ArXiv search
        </a>
        <span className="connection"><i />{status}</span>
      </header>
      {(embeddingBusy || indexJobProgress || indexJobMessage) && (
        <div className="embedding-progress" aria-live="polite">
          <div className="progress-heading">
            <strong>{indexJobProgress?.phase ?? (embeddingBusy ? "Indexing papers" : "Index job")}</strong>
            {indexJobProgress?.total_papers_known && (
              <span>
                {indexJobProgress.scanned_papers.toLocaleString()} /{" "}
                {indexJobProgress.total_papers.toLocaleString()} papers
              </span>
            )}
          </div>
          <progress
            aria-label="Backend indexing progress"
            max={Math.max(indexJobProgress?.phase_total ?? indexJobProgress?.total_papers ?? 1, 1)}
            value={
              indexJobProgress?.phase_total != null
                ? indexJobProgress.phase_completed
                : indexJobProgress?.total_papers_known
                  ? (embeddingBusy && indexJobProgress.scanned_papers >= indexJobProgress.total_papers
                    ? undefined
                    : indexJobProgress.scanned_papers)
                  : undefined
            }
          />
          {indexJobMessage && <span role="status">{indexJobMessage}</span>}
          {indexJobMessage && indexJobSubscriptionRef.current && (
            <button
              className="chunk-button"
              onClick={() => {
                const subscription = indexJobSubscriptionRef.current;
                if (subscription) connectIndexingProgress(subscription);
              }}
            >
              Reconnect progress
            </button>
          )}
          {indexJobProgress && <>
          <span>
            Index job {indexJobProgress.state}
            {indexJobProgress.phase_total != null && (
              <> · {indexJobProgress.phase_completed.toLocaleString()} / {indexJobProgress.phase_total.toLocaleString()} rows prepared</>
            )}
            {indexJobProgress.total_papers_known && (
              <> · {indexJobProgress.scanned_papers.toLocaleString()} / {indexJobProgress.total_papers.toLocaleString()} papers</>
            )}
            {" · "}
            {indexJobProgress.papers_per_second.toFixed(1)} papers/s · ETA{" "}
            {formatDuration(indexJobProgress.eta_seconds)} · elapsed{" "}
            {formatDuration(indexJobProgress.elapsed_seconds)}
            {indexJobProgress.chunked_papers.toLocaleString()} chunked ·{" "}
            {indexJobProgress.embedded_papers.toLocaleString()} embedded ·{" "}
            {indexJobProgress.failed_papers.toLocaleString()} failed
          </span>
          {indexJobProgress.error && <span role="alert">Latest error: {indexJobProgress.error}</span>}
          </>}
        </div>
      )}

      {!selected && (
        <>
      <section className="hero">
        <h1>Paper search</h1>
        <form className="search-form" onSubmit={search}>
          <input
            aria-label="Search paper text"
            autoFocus
            placeholder="Search paper text"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
          <button
            type="submit"
            disabled={busy || (!lexicalSearch && !semanticSearch) || (!query.trim() && !semanticSearch)}
          >
            {busy ? "Searching…" : "Search"}
            <span aria-hidden="true">↗</span>
          </button>
        </form>
        <div className="search-note">
          <span>Local index · Search results include matching text and metadata</span>
          <label className="render-toggle">
            <input
              type="checkbox"
              checked={lexicalSearch}
              onChange={(event) => setLexicalSearch(event.target.checked)}
            />
            Lexical
          </label>
          <label className="render-toggle">
            <input
              type="checkbox"
              checked={semanticSearch}
              onChange={(event) => setSemanticSearch(event.target.checked)}
            />
            Semantic
          </label>
          <label className="render-toggle planned-toggle" title="Reranking is planned and not available yet">
            <input type="checkbox" disabled checked={false} />
            Rerank (planned)
          </label>
          <label className="model-select-label">
            Embedding mode
            <select
              value={embeddingMode}
              onChange={(event) => setEmbeddingMode(event.target.value as EmbeddingMode)}
              disabled={embeddingBusy || queryPipelineBusy}
            >
              <option value="full_text">Full-text chunks</option>
              <option value="abstract">Abstract only</option>
            </select>
          </label>
          <label className="render-toggle">
            <input
              type="checkbox"
              checked={renderLatex}
              onChange={(event) => setRenderLatex(event.target.checked)}
            />
            Render LaTeX
          </label>
          <button
            className="chunk-button"
            onClick={() => {
              if (!indexJobProgress || !["running", "paused", "queued"].includes(indexJobProgress.state)) {
                void startFullIndexingJob();
              } else {
                void indexOrResume();
              }
            }}
            disabled={indexJobControlBusy || (!indexJobProgress && (embeddingBusy || !selectedEmbeddingModel))}
          >
            {indexJobControlBusy
              ? "Updating…"
              : indexJobProgress?.state === "paused"
                ? "Resume"
                : indexJobProgress?.state === "running"
                  ? "Pause"
                  : embeddingBusy
                    ? "Embedding…"
                    : "Embed"}
          </button>
          {abstractEmbeddingIndexStatus?.has_embeddings &&
            (!abstractEmbeddingIndexStatus.vector_index_ready ||
              abstractEmbeddingIndexStatus.unindexed_vectors > 0 ||
              abstractEmbeddingIndexStatus.index_unindexed_rows > 0) && (
              <button
                className="chunk-button"
                onClick={() => void buildAbstractVectorIndex()}
                disabled={abstractIndexBusy || embeddingBusy}
              >
                {abstractIndexBusy ? "Building index…" : "Update vector index"}
              </button>
            )}
          {semanticSearch && abstractIndexError && (
            <span role="alert">{abstractIndexError}</span>
          )}
        </div>
      </section>

      <section className="results-section" aria-live="polite">
        {error && <div className="error-box">{error}</div>}
        {hits.length > 0 && (
          <div className="results-heading">
            <div className="results-summary">
              <h2>{hits.length} matching papers</h2>
              {lookupTimeMs !== null && <span className="lookup-time">{lookupTimeMs} ms</span>}
            </div>
            <div className="results-actions">
              <span className="query-chip">{query}</span>
              {searchUrl && <a className="results-link" href={searchUrl}>Share results <span aria-hidden="true">↗</span></a>}
            </div>
          </div>
        )}
        <div className="results-grid">
          <div className="result-list">
            {hits.map((hit, index) => (
              <article className="paper-card" key={hit.paper_id}>
                <div className="card-topline">
                  <span className="result-number">{index + 1}.</span>
                  <button className="paper-id" onClick={() => void openPaper(hit.paper_id)}>{hit.paper_id} ↗</button>
                </div>
                <button className="paper-title" onClick={() => void openPaper(hit.paper_id)}>
                  {hit.metadata?.title || "Untitled paper"}
                </button>
                <p className="authors">{hit.metadata?.authors || "Author information unavailable"}</p>
                {hit.metadata?.abstract && (
                  <p className="abstract"><LatexSnippet source={hit.metadata.abstract} query={query} renderLatex={renderLatex} /></p>
                )}
                <div className="snippet"><LatexSnippet source={hit.snippet} query={query} renderLatex={renderLatex} /></div>
                <div className="card-footer">
                  <span className="category">Category: {hit.metadata?.primary_category || hit.metadata?.categories || "unavailable"}</span>
                  {hit.metadata?.journal_ref && <span>Journal: {hit.metadata.journal_ref}</span>}
                  {hit.metadata?.doi && (
                    <a href={`https://doi.org/${encodeURIComponent(hit.metadata.doi)}`} target="_blank" rel="noreferrer">
                      DOI: {hit.metadata.doi}
                    </a>
                  )}
                  {!hit.metadata?.doi && <span>DOI: unavailable</span>}
                  {hit.metadata?.license && (
                    <a href={hit.metadata.license} target="_blank" rel="noreferrer">License</a>
                  )}
                  <a href={`https://arxiv.org/abs/${encodeURIComponent(hit.paper_id)}`} target="_blank" rel="noreferrer">arXiv record</a>
                </div>
              </article>
            ))}
            {hasMore && (
              <button className="load-more" onClick={() => void loadMore()} disabled={busy}>
                {busy ? "Loading…" : "Load more"}
              </button>
            )}
            {!hits.length && !busy && !error && (
              <div className="empty-state">
                <div className="empty-symbol">⌘</div>
                <h2>Your next useful paper is in here.</h2>
                <p>Search the local full-text index to see paper details and matching text.</p>
              </div>
            )}
          </div>
        </div>
      </section>
        </>
      )}
      {selected && (
        <section className="paper-reader" aria-labelledby="reader-title">
            <div className="reader-header">
              <div>
                <span className="eyebrow">{renderLatex ? "PAPER · " : "PAPER SOURCE · "}{selected.paper_id}</span>
                <h2 id="reader-title">{selected.metadata?.title || selected.paper_id}</h2>
              </div>
              <button className="back-to-results" onClick={closePaper}>
                ← {hits.length ? "Back to results" : "Back to search"}
              </button>
            </div>
            <div className="reader-meta">
              {selected.metadata?.authors && <span>{selected.metadata.authors}</span>}
              {selected.metadata?.primary_category && <span>{selected.metadata.primary_category}</span>}
              {selected.metadata?.journal_ref && <span>{selected.metadata.journal_ref}</span>}
              {selected.metadata?.doi ? (
                <a href={`https://doi.org/${encodeURIComponent(selected.metadata.doi)}`} target="_blank" rel="noreferrer">
                  DOI: {selected.metadata.doi}
                </a>
              ) : (
                <span>DOI unavailable</span>
              )}
              {selected.metadata?.license && (
                <a href={selected.metadata.license} target="_blank" rel="noreferrer">License</a>
              )}
            </div>
            <div className="chunk-actions">
              <label className="model-select-label">
                Embedding model
                <select
                  value={selectedEmbeddingModel}
                  onChange={(event) => {
                    setSelectedEmbeddingModel(event.target.value);
                    setEmbeddingModel("");
                    setChunkResult((current) => current && ({
                      ...current,
                      chunks: current.chunks.map((chunk) => ({
                        ...chunk,
                        embeddingStatus: "pending",
                        embeddingError: undefined,
                      })),
                    }));
                    setQueryPipelineResult(null);
                  }}
                  disabled={!modelChoices.embedding.length || embeddingBusy || queryPipelineBusy}
                >
                  {!modelChoices.embedding.length && <option value="">No embedding models available</option>}
                  {modelChoices.embedding.map((model) => (
                    <option key={model.id} value={model.id}>{model.id}</option>
                  ))}
                </select>
              </label>
              <label className="model-select-label">
                Reranker model
                <select
                  value={selectedRerankerModel}
                  onChange={(event) => setSelectedRerankerModel(event.target.value)}
                  disabled={!modelChoices.reranking.length || embeddingBusy || queryPipelineBusy}
                >
                  {!modelChoices.reranking.length && <option value="">No tagged reranker models</option>}
                  {modelChoices.reranking.map((model) => (
                    <option key={model.id} value={model.id}>{model.id}</option>
                  ))}
                </select>
              </label>
              <label className="chunk-budget-label">
                Estimated tokens
                <input
                  type="number"
                  min={1}
                  max={2048}
                  step={64}
                  value={tokenBudget}
                  onChange={(event) => setTokenBudget(Number(event.target.value))}
                />
              </label>
              <button
                className="chunk-button"
                onClick={() => void chunkSelectedPaper()}
                disabled={chunkBusy || embeddingBusy || tokenBudget < 1 || tokenBudget > 2048}
              >
                {chunkBusy ? "Chunking…" : chunkResult ? "Re-chunk paper" : "Chunk paper"}
              </button>
              <button
                className="chunk-button"
                onClick={() => void indexSelectedPaper()}
                disabled={
                  chunkBusy ||
                  !selectedEmbeddingModel ||
                  (embeddingMode === "full_text" &&
                    (!chunkResult ||
                      chunkResult.chunks.length === 0 ||
                      chunkResult.chunks.every((chunk) => chunk.embeddingStatus === "done")))
                }
              >
                {embeddingBusy ? "Embedding…" : "Embed"}
              </button>
              {chunkView && (
                <button className="chunk-button secondary" onClick={() => setChunkView(false)}>
                  Show paper
                </button>
              )}
              {chunkResult && !chunkView && (
                <button className="chunk-button secondary" onClick={() => setChunkView(true)}>
                  Show chunks
                </button>
              )}
            </div>
            <div className="query-pipeline-controls">
              <label className="model-select-label">
                Embedding mode
                <select
                  value={embeddingMode}
                  onChange={(event) => setEmbeddingMode(event.target.value as EmbeddingMode)}
                  disabled={embeddingBusy || queryPipelineBusy}
                >
                  <option value="full_text">Full-text chunks</option>
                  <option value="abstract">Abstract only</option>
                </select>
              </label>
              <label className="chunk-query-label">
                Test hybrid retrieval
                <input
                  className="chunk-query"
                  value={rerankQuery}
                  onChange={(event) => setRerankQuery(event.target.value)}
                  placeholder="Enter a query to test lexical + vector retrieval"
                  maxLength={2000}
                />
              </label>
              <button
                className="chunk-button"
                onClick={() => void testQueryPipeline()}
                disabled={
                  queryPipelineBusy ||
                  !rerankQuery.trim() ||
                  !selectedEmbeddingModel ||
                  (embeddingMode === "full_text" &&
                    (!chunkResult ||
                      chunkResult.chunks.every((chunk) => chunk.embeddingStatus !== "done")))
                }
              >
                {queryPipelineBusy ? "Searching and reranking…" : "Test hybrid search"}
              </button>
            </div>
            {modelChoicesError && <div className="error-box">{modelChoicesError}</div>}
            {queryPipelineResult && (
              <section className="query-pipeline-results" aria-label="Query embedding and reranking results">
                <div className="chunk-summary">
                  <strong>Top {queryPipelineResult.hits.length} fused candidates</strong>
                  <span>
                    Embedding: {queryPipelineResult.embedding_model}
                    {" · "}Reranker: {queryPipelineResult.reranker_model ?? "not configured"}
                  </span>
                </div>
                <div className="chunk-list">
                  {queryPipelineResult.hits.map((hit, index) => (
                    <article className="chunk-card" key={hit.chunk_id}>
                      <div className="chunk-card-heading">
                        <strong>Rank {index + 1} · fused {hit.fused_score.toFixed(5)}</strong>
                        <span>
                          {hit.lexical_rank !== null && `lexical #${hit.lexical_rank}`}
                          {hit.semantic_rank !== null && ` · dense #${hit.semantic_rank}`}
                          {hit.rerank_score !== null &&
                            ` · rerank ${
                              Math.abs(hit.rerank_score) < 0.0001
                                ? hit.rerank_score.toExponential(2)
                                : hit.rerank_score.toFixed(4)
                            }`}
                        </span>
                      </div>
                      <pre>{hit.text}</pre>
                    </article>
                  ))}
                </div>
              </section>
            )}
            {chunkError && <div className="error-box">{chunkError}</div>}
            {chunkView && chunkResult && (
              <section className="chunk-view" aria-label="Paper chunk preview">
                <div className="chunk-summary">
                  <strong>{chunkResult.chunks.length} chunks · {chunkResult.source_chars.toLocaleString()} source characters</strong>
                  {embeddingModel && <span>Embedding: {embeddingModel}</span>}
                </div>
                {chunkResult.source_truncated && (
                  <p className="truncation-note">Only the first 1,000,000 characters were chunked.</p>
                )}
                <div className="chunk-list">
                  {chunkResult.chunks.map((chunk, index) => (
                    <article
                      className={`chunk-card status-${chunk.embeddingStatus}`}
                      key={`${chunk.byte_range[0]}-${chunk.byte_range[1]}`}
                    >
                      <div className="chunk-card-heading">
                        <strong>Chunk {index + 1}</strong>
                        <span>
                          {chunk.cost} estimated tokens · {chunk.word_count} words
                          {" · "}bytes {chunk.byte_range[0].toLocaleString()}–{chunk.byte_range[1].toLocaleString()}
                        </span>
                      </div>
                      <div className={`chunk-embedding-status ${chunk.embeddingStatus}`}>
                        {chunk.embeddingStatus === "pending" && "Not embedded"}
                        {chunk.embeddingStatus === "embedding" && "Embedding…"}
                        {chunk.embeddingStatus === "done" && "Done"}
                        {chunk.embeddingStatus === "failed" && "Failed"}
                        {chunk.embeddingError && ` · ${chunk.embeddingError}`}
                      </div>
                      <pre>{chunk.text}</pre>
                    </article>
                  ))}
                </div>
              </section>
            )}
            {!chunkView && <PaperRenderer source={selected.text} renderLatex={renderLatex} />}
            {selected.truncated && (
              <p className="truncation-note">
                This paper exceeds the 1,000,000-character reader limit; showing the first {selected.returned_chars.toLocaleString()} characters.
              </p>
            )}
        </section>
      )}
      <footer className="footer">A local research tool · Search results are limited to the data currently indexed</footer>
    </main>
  );
}

export default App;
