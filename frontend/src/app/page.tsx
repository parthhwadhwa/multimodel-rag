"use client";

import { useState } from "react";
import Header from "@/components/Header";
import ChatInput from "@/components/ChatInput";
import ResponseCard from "@/components/ResponseCard";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Home() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");
  const [modelUsed, setModelUsed] = useState<string>("");
  const [citations, setCitations] = useState<any[]>([]);
  const [confidenceScore, setConfidenceScore] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isError, setIsError] = useState(false);

  const handleSend = async (text: string) => {
    setQuery(text);
    setResponse("");
    setModelUsed("");
    setCitations([]);
    setConfidenceScore(null);
    setIsError(false);
    setIsLoading(true);

    try {
      const res = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text, model: "ollama" }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => null);
        const detail = errData?.detail || `Request failed (HTTP ${res.status})`;
        setResponse(detail);
        setModelUsed("");
        setIsError(true);
        return;
      }

      const data = await res.json();
      setResponse(data.answer);
      setModelUsed(data.model || data.model_used);
      setCitations(data.citations || []);
      setConfidenceScore(data.confidence_score !== undefined ? data.confidence_score : null);
    } catch (error) {
      console.error("API Error:", error);
      setResponse("Could not reach the PharmaGuide API. Make sure the server is running.");
      setIsError(true);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-[#0f172a] text-[#f8fafc] flex flex-col items-center selection:bg-slate-700">

      {/* Top Header */}
      <header className="w-full flex flex-col items-center py-6 bg-[#0f172a]/90 backdrop-blur-md border-b border-[#334155] z-20 shrink-0">
        <Header />
      </header>

      {/* Main Chat Area */}
      <div className="flex-1 w-full max-w-[800px] overflow-y-auto px-4 md:px-6 pt-8 pb-32 flex flex-col no-scrollbar">
        {!query && !response && !isLoading && !isError ? (
          <div className="flex flex-col items-center justify-center flex-1 h-full animate-[slideUpFade_0.6s_ease-out_forwards] gap-8 mt-12">

            <div className="flex flex-col items-center gap-4 text-center mt-4">
              <p className="text-slate-400 font-medium text-[15px]">Try asking:</p>
              <div className="flex flex-col gap-3 w-full max-w-[400px]">
                {[
                  "What is albuterol used for?",
                  "What are the side effects of ibuprofen?",
                  "What dosage of metformin is recommended?"
                ].map((example, i) => (
                  <button
                    key={i}
                    onClick={() => handleSend(example)}
                    className="px-5 py-3.5 bg-[#1e293b] border border-[#334155] rounded-xl text-[14px] text-[#f8fafc] hover:border-slate-400 hover:shadow-sm transition-all focus:outline-none focus:ring-2 focus:ring-slate-500 text-left"
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>

          </div>
        ) : (
          <div className="w-full flex flex-col gap-8">
            <ResponseCard
              query={query}
              response={response}
              modelUsed={modelUsed}
              citations={citations}
              confidenceScore={confidenceScore}
              isLoading={isLoading}
              isError={isError}
            />
          </div>
        )}
      </div>

      {/* Fixed Bottom Input */}
      <div className="fixed bottom-0 left-0 right-0 p-4 md:p-6 bg-gradient-to-t from-[#0f172a] via-[#0f172a]/95 to-transparent z-20 pointer-events-none flex justify-center">
        <div className="w-full max-w-[800px] pointer-events-auto shadow-[0_0_40px_rgba(15,23,42,1)]">
          <ChatInput onSend={handleSend} isLoading={isLoading} />
        </div>
      </div>
    </main>
  );
}
