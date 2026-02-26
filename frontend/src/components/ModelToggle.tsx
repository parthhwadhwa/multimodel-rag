"use client";

import React from "react";

type Model = "ollama" | "gemini";

const IS_PRODUCTION = process.env.NEXT_PUBLIC_IS_PRODUCTION === "true";

interface ModelToggleProps {
    selected: Model;
    onChange: (model: Model) => void;
    disabled?: boolean;
}

export default function ModelToggle({ selected, onChange, disabled }: ModelToggleProps) {
    return (
        <div
            className={`relative inline-flex items-center bg-neutral-100/60 backdrop-blur-xl p-1 rounded-full border border-black/[0.04] shadow-[0_2px_8px_inset_rgba(0,0,0,0.02)] transition-opacity duration-300 ${disabled ? 'opacity-50 pointer-events-none' : 'opacity-100'}`}
            role="radiogroup"
            aria-label="Select AI Model"
        >
            {/* Sliding Underlay */}
            <div
                className="absolute h-[calc(100%-8px)] rounded-full bg-white shadow-[0_1px_3px_rgba(0,0,0,0.08),0_1px_2px_rgba(0,0,0,0.04)] transition-all duration-300 ease-[cubic-bezier(0.23,1,0.32,1)]"
                style={{
                    width: "100px",
                    left: selected === "ollama" ? "4px" : "108px"
                }}
            />

            {/* Ollama Option */}
            <button
                type="button"
                role="radio"
                aria-checked={selected === "ollama"}
                onClick={() => onChange("ollama")}
                title={IS_PRODUCTION ? "Ollama is unavailable in the deployed environment" : undefined}
                className={`relative z-10 w-[100px] py-1.5 text-[13px] font-medium transition-colors duration-200 rounded-full focus:outline-none focus-visible:ring-2 focus-visible:ring-neutral-400 ${IS_PRODUCTION
                        ? "text-neutral-400 cursor-not-allowed"
                        : selected === "ollama" ? "text-neutral-900" : "text-neutral-500 hover:text-neutral-700"
                    }`}
            >
                {IS_PRODUCTION ? (
                    <span className="flex flex-col items-center leading-tight">
                        <span>Local (Ollama)</span>
                        <span className="text-[9px] font-normal text-amber-500 tracking-wide">unavailable</span>
                    </span>
                ) : "Local (Ollama)"}
            </button>

            {/* Gemini Option */}
            <button
                type="button"
                role="radio"
                aria-checked={selected === "gemini"}
                onClick={() => onChange("gemini")}
                className={`relative z-10 w-[100px] py-1.5 text-[13px] font-medium transition-colors duration-200 rounded-full focus:outline-none focus-visible:ring-2 focus-visible:ring-neutral-400 ${selected === "gemini" ? "text-neutral-900" : "text-neutral-500 hover:text-neutral-700"
                    }`}
            >
                Cloud (Gemini)
            </button>
        </div>
    );
}
