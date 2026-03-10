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
        onSend(val.trim());
        setVal("");
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
        }
    };

    useEffect(() => {
        inputRef.current?.focus();
    }, []);

    return (
        <form
            onSubmit={handleSubmit}
            style={{
                display: "flex",
                gap: "8px",
                padding: "16px 24px",
                background: "var(--header-bg)",
                borderTop: "1px solid var(--border)",
            }}
        >
            <input
                ref={inputRef}
                type="text"
                value={val}
                onChange={(e) => setVal(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={isLoading}
                placeholder="Ask about any medication (e.g., side effects of metformin)"
                style={{
                    flex: 1,
                    padding: "12px 16px",
                    fontSize: "14px",
                    border: "1px solid var(--border)",
                    borderRadius: "8px",
                    outline: "none",
                    background: "var(--bg)",
                    color: "var(--text-primary)",
                }}
                aria-label="Ask about medication"
            />
            <button
                type="submit"
                disabled={!val.trim() || isLoading}
                style={{
                    padding: "12px 24px",
                    fontSize: "14px",
                    fontWeight: 600,
                    color: "#ffffff",
                    background: !val.trim() || isLoading ? "#93c5fd" : "var(--user-bubble)",
                    border: "none",
                    borderRadius: "8px",
                    cursor: !val.trim() || isLoading ? "not-allowed" : "pointer",
                    transition: "background 0.2s",
                }}
                aria-label="Send"
            >
                Send
            </button>
        </form>
    );
}
