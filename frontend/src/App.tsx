import { useMemo, useRef, useState, type ChangeEvent } from "react";
import { askQuestion } from "./client";
import type { AskRequest, AskResponse, SourceChunk } from "./types";

const DEFAULT_REQUEST: AskRequest = {
  question: "Trước khi có chữ Nôm, người Việt có từng có chữ viết riêng không?",
  top_k: 10,
  pool_size: 20,
  rerank: true,
};

export default function App() {
  const [request, setRequest] = useState<AskRequest>(DEFAULT_REQUEST);
  const [answer, setAnswer] = useState<AskResponse | null>(null);
  const [selectedSource, setSelectedSource] = useState<SourceChunk | null>(
    null
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedKey, setExpandedKey] = useState<string | null>(null);
  const [answerHighlighted, setAnswerHighlighted] = useState(false);
  const answerRef = useRef<HTMLDivElement | null>(null);
  const answerTextRef = useRef<HTMLDivElement | null>(null);

  const resolvedSource = useMemo(() => {
    if (selectedSource) {
      return selectedSource;
    }
    return answer?.sources?.[0] ?? null;
  }, [answer?.sources, selectedSource]);

  const resolvedSourceLabel = useMemo(() => {
    if (!resolvedSource) {
      return "Chưa chọn nguồn";
    }
    const pageSuffix = resolvedSource.page_number
      ? ` – trang ${resolvedSource.page_number}`
      : "";
    return `${resolvedSource.label}${pageSuffix}`;
  }, [resolvedSource]);

  const scrollAnswerIntoView = () => {
    if (typeof window === "undefined") {
      return;
    }
    window.setTimeout(() => {
      const target = answerTextRef.current ?? answerRef.current;
      target?.scrollIntoView({ behavior: "smooth", block: "center" });
    }, 0);
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await askQuestion(request);
      setAnswer(response);
      setSelectedSource(response.sources?.[0] ?? null);
      setExpandedKey(null);
      setAnswerHighlighted(true);
      scrollAnswerIntoView();
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to reach backend";
      setError(message);
      setAnswer(null);
      setAnswerHighlighted(false);
    } finally {
      setLoading(false);
    }
  };

  const handleSourceClick = (source: SourceChunk) => {
    setSelectedSource(source);
    setAnswerHighlighted(false);
  };

  const clearAnswerHighlight = () => setAnswerHighlighted(false);

  const handleInputChange = (evt: ChangeEvent<HTMLTextAreaElement>) => {
    const value = evt.target.value;
    setRequest((prev: AskRequest) => ({ ...prev, question: value }));
    clearAnswerHighlight();
  };

  return (
    <div className="app-wrapper">
      <div className="cloud-layer" aria-hidden="true">
        <span className="cloud cloud-1" />
        <span className="cloud cloud-2" />
      </div>
      <div className="app-shell">
        <section className="hero-card">
          <h1 className="hero-title">Nom-sense</h1>
          <p className="hero-subtitle">
            Hỏi đáp với Gs. Nguyễn Quang Hồng về chữ Nôm.
          </p>
        </section>

        <section className="panel">
          <div>
            <label htmlFor="question">Câu hỏi</label>
            <textarea
              id="question"
              value={request.question}
              onChange={handleInputChange}
              placeholder="Ví dụ: Ai là người đầu tiên được cho là làm thơ Nôm?"
            />
          </div>

          <button
            type="button"
            onClick={handleSubmit}
            disabled={loading}
            className={loading ? "primary-button is-loading" : "primary-button"}
          >
            {!loading && <span aria-hidden="true" className="button-icon" />}
            <span>{loading ? "Đang phân tích…" : "Đặt câu hỏi"}</span>
          </button>

          {loading && (
            <div className="loading-indicator" role="status" aria-live="polite">
              <span className="spinner" aria-hidden="true" />
              <span>Chờ Gs. Nguyễn Quang Hồng giải đáp…</span>
            </div>
          )}

          {error && <p className="error-text">{error}</p>}

          {answer && (
            <div
              className={
                answerHighlighted
                  ? "answer-card answer-card--highlight"
                  : "answer-card"
              }
              ref={answerRef}
              onClick={clearAnswerHighlight}
              onFocusCapture={clearAnswerHighlight}
            >
              <h2>Kết quả</h2>
              <div className="answer-text" ref={answerTextRef}>
                {answer.answer}
              </div>
              {answer.sources && answer.sources.length > 0 && (
                <div className="citation-list">
                  <strong>Nguồn tham chiếu</strong>
                  {answer.sources.map((source: SourceChunk, idx: number) => {
                    const isActive =
                      (resolvedSource?.viewer_url === source.viewer_url &&
                        resolvedSource?.label === source.label) ||
                      selectedSource?.label === source.label;
                    const key = `${source.file_name ?? source.label}-${
                      source.page_number ?? idx
                    }`;
                    const isExpanded = expandedKey === key;
                    const titleParts: string[] = [];
                    const bookTitle = source.book_title?.trim();
                    const cleanedChapter = source.chapter
                      ? source.chapter
                          .replace(/_/g, " ")
                          .replace(/\s+/g, " ")
                          .trim()
                      : "";
                    const hideChapter =
                      bookTitle === "Ngôn ngữ. Văn tự. Ngữ văn (Tuyển tập)";
                    if (bookTitle) {
                      titleParts.push(bookTitle);
                    }
                    if (
                      cleanedChapter &&
                      !hideChapter &&
                      cleanedChapter !== bookTitle &&
                      !titleParts.includes(cleanedChapter)
                    ) {
                      titleParts.push(cleanedChapter);
                    }
                    const titleText =
                      titleParts.length > 0
                        ? titleParts.join(" – ")
                        : source.label;
                    return (
                      <article
                        key={key}
                        className={`citation-item${isActive ? " active" : ""}${
                          isExpanded ? " expanded" : ""
                        }`}
                      >
                        <button
                          type="button"
                          className="citation-toggle"
                          onClick={() => {
                            handleSourceClick(source);
                            clearAnswerHighlight();
                            setExpandedKey(isExpanded ? null : key);
                          }}
                          aria-expanded={isExpanded}
                        >
                          <div className="citation-meta">
                            <span className="citation-title">{titleText}</span>
                            {source.page_number && (
                              <span className="citation-page">
                                P.{source.page_number}
                              </span>
                            )}
                          </div>
                          <span
                            className={
                              isExpanded
                                ? "citation-icon citation-icon--open"
                                : "citation-icon"
                            }
                            aria-hidden="true"
                          />
                        </button>
                        {isExpanded && (
                          <div className="snippet">{source.text}</div>
                        )}
                      </article>
                    );
                  })}
                </div>
              )}
            </div>
          )}
        </section>

        <footer className="site-footer">
          <p className="footer-line">
            Tư liệu trích từ Khái luận Văn tự học chữ Nôm, Nguyễn Quang Hồng,
            NXB Giáo dục, 2008 & Ngôn ngữ. Văn tự. Ngữ văn, Nguyễn Quang Hồng,
            NXB Khoa học Xã hội, 2018 .
          </p>
          <p className="footer-line">
            Source: An Introduction to the Study of Nôm Script (Khái luận Văn tự
            học chữ Nôm), Nguyễn Quang Hồng, Education Publishing House, 2008,
            and Language. Script. Literature (Ngôn ngữ. Văn tự. Ngữ văn), Nguyễn
            Quang Hồng, Social Sciences Publishing House, 2018.
          </p>
          <span className="footer-divider" aria-hidden="true" />
          <p className="footer-line">© 2025 Digitizing Việt Nam</p>
        </footer>
      </div>
    </div>
  );
}
