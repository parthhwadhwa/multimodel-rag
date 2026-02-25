import React from "react";

export default function Header() {
    return (
        <header className="w-full py-12 flex flex-col items-center justify-center fade-in">
            <h1 className="text-4xl md:text-5xl font-semibold tracking-tight text-neutral-900 mb-3">
                MediRAG
            </h1>
            <p className="text-[15px] text-neutral-500 font-medium tracking-wide">
                Grounded Drug Information. Zero Hallucination.
            </p>
        </header>
    );
}
