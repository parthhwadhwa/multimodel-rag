"use client";

import { useState } from "react";
import Header from "@/components/Header";
import ModelToggle from "@/components/ModelToggle";
import ChatInput from "@/components/ChatInput";
import ResponseCard from "@/components/ResponseCard";

type Model = "ollama" | "gemini";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const IS_PRODUCTION = process.env.NEXT_PUBLIC_IS_PRODUCTION === "true";

export default function Home() {
  const [model, setModel] = useState<Model>(IS_PRODUCTION ? "gemini" : "ollama");
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");
  const [modelUsed, setModelUsed] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);
  const [isError, setIsError] = useState(false);

  const handleSend = async (text: string) => {
    setQuery(text);
    setResponse("");
    setModelUsed("");
    setIsError(false);
    setIsLoading(true);

    try {
      const res = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text, model: model }),
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
      setModelUsed(data.model_used);
    } catch (error) {
      console.error("API Error:", error);
      setResponse("Could not reach the MediRAG API. Make sure the server is running.");
      setIsError(true);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-[#fafafc] bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-white via-[#fafafc] to-[#fafafc] flex flex-col items-center selection:bg-neutral-200 text-neutral-900">

      {/* Spacer for vertical balance */}
      <div className="w-full h-[15vh] min-h-[80px]" />

      <div className="w-full max-w-3xl px-4 md:px-6 flex flex-col items-center flex-1">

        {/* Header & Toggle shrink when a response is loaded to keep focus on content */}
        <div className={`flex flex-col items-center w-full transition-all duration-700 ease-[cubic-bezier(0.23,1,0.32,1)] origin-top ${query ? 'opacity-0 h-0 overflow-hidden scale-95 pointer-events-none' : 'opacity-100 h-auto mb-8 scale-100'}`}>
          <Header />
          <div className="mt-2 mb-4">
            <ModelToggle
              selected={model}
              onChange={setModel}
              disabled={isLoading}
            />
          </div>
        </div>

        {/* The Search Bar Input */}
        <div className={`w-full transition-all duration-700 ease-[cubic-bezier(0.23,1,0.32,1)] z-10 ${query ? 'sticky top-6 rounded-2xl shadow-xl' : ''}`}>
          <ChatInput onSend={handleSend} isLoading={isLoading} />
        </div>

        {/* Global minimal custom animation injected directly for ease of use without heavy Tailwind setup */}
        <style jsx global>{`
          @keyframes slideUpFade {
            0% {
              opacity: 0;
              transform: translateY(16px);
            }
            100% {
              opacity: 1;
              transform: translateY(0);
            }
          }
          .animate-[slideUpFade_0.6s_ease-out_forwards] {
            animation: slideUpFade 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
          }
          /* Hide scrollbar for a cleaner feel but keep functionality */
          ::-webkit-scrollbar {
            width: 0px;
            background: transparent;
          }
        `}</style>

        {/* The Output Flow */}
        <div className="w-full pb-24 relative z-0">
          <ResponseCard
            query={query}
            response={response}
            modelUsed={modelUsed}
            isLoading={isLoading}
            isError={isError}
          />
        </div>

      </div>
    </main>
  );
}
