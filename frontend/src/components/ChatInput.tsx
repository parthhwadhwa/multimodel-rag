"use client";

import React, { useState, useRef, useEffect } from "react";

interface ChatInputProps {
    onSend: (query: string) => void;
    isLoading: boolean;
}

export default function ChatInput({ onSend, isLoading }: ChatInputProps) {
    const [val, setVal] = useState("");
    const inputRef = useRef<HTMLInputElement>(null);

    const handleSubmit = (e?: React.FormEvent) => {
        if (e) e.preventDefault();
        if (!val.trim() || isLoading) return;
        onSend(val);
        setVal("");
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
        }
    };

    // Auto-focus on mount
    useEffect(() => {
        inputRef.current?.focus();
    }, []);

    return (
        <form
            onSubmit={handleSubmit}
            className={`w-full max-w-2xl mx-auto relative group transition-all duration-300 ${isLoading ? "opacity-70 pointer-events-none" : "hover:scale-[1.005] focus-within:scale-[1.01]"}`}
        >
            <div className="absolute inset-y-0 left-4 flex items-center pointer-events-none">
                <svg className="w-5 h-5 text-neutral-400 group-focus-within:text-neutral-500 transition-colors duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
            </div>

            <input
                ref={inputRef}
                type="text"
                value={val}
                onChange={(e) => setVal(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={isLoading}
                placeholder="Ask about a medication, side effects, or contraindications..."
                className="w-full py-4 pl-12 pr-14 bg-white/80 backdrop-blur-md border border-neutral-200/60 rounded-2xl shadow-[0_8px_30px_rgb(0,0,0,0.04)] text-[15px] text-neutral-800 placeholder:text-neutral-400 focus:outline-none focus:ring-[3px] focus:ring-slate-100 focus:border-slate-300 transition-all duration-300 disabled:bg-neutral-50/50"
                aria-label="Search medication information"
            />

            <div className="absolute inset-y-0 right-2 flex items-center">
                <button
                    type="submit"
                    disabled={!val.trim() || isLoading}
                    className="p-2 mr-1 rounded-xl text-neutral-400 hover:text-neutral-800 hover:bg-neutral-100/50 disabled:opacity-30 disabled:hover:text-neutral-400 disabled:hover:bg-transparent transition-all duration-300 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-200"
                    aria-label="Send Query"
                >
                    {isLoading ? (
                        <svg className="animate-spin w-5 h-5 text-neutral-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                    ) : (
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4.5 12h15m0 0l-6.75-6.75M19.5 12l-6.75 6.75" />
                        </svg>
                    )}
                </button>
            </div>
        </form>
    );
}
