import React from "react";

interface MetadataBarProps {
    modelUsed?: string;
}

export default function MetadataBar({ modelUsed }: MetadataBarProps) {
    if (!modelUsed) return null;

    return (
        <div className="mt-8 pt-5 flex flex-col md:flex-row md:items-center gap-2 md:gap-4 text-[11px] font-medium tracking-wide text-neutral-400 border-t border-neutral-100">
            <div className="flex items-center gap-1.5">
                <span className="text-neutral-400/80">Source:</span>
                <span className="text-neutral-500">MediRAG System</span>
            </div>

            <div className="hidden md:block w-1 h-1 rounded-full bg-neutral-200"></div>

            <div className="flex items-center gap-1.5 uppercase">
                <span className="text-neutral-400/80">Model:</span>
                <span className="text-neutral-500 flex items-center gap-1.5">
                    {modelUsed}
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400/80 shadow-[0_0_8px_rgba(52,211,153,0.4)]"></span>
                </span>
            </div>
        </div>
    );
}
