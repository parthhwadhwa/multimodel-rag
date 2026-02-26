import React from "react";
import LoadingCard from "./LoadingCard";
import MetadataBar from "./MetadataBar";

interface ResponseCardProps {
    query: string;
    response: string;
    modelUsed?: string;
    isLoading?: boolean;
    isError?: boolean;
}

// Simple text formatter to handle basic markdown-like structures
const FormattedResponse = ({ text }: { text: string }) => {
    // Basic markdown list normalization: handles '1. ', '- ', '* '
    const normalizedText = text.replace(/^(?:\s*)(?:[-*]|\d+\.)\s+(.+)$/gm, '• $1');
    const sections = normalizedText.split('\n\n').filter(Boolean);

    return (
        <div className="flex flex-col gap-5 text-[15px] leading-relaxed text-neutral-700 max-w-[70ch]">
            {sections.map((section, idx) => {
                // If section is a list
                if (section.trim().includes('• ')) {
                    const items = section.split('\n').filter(Boolean);
                    return (
                        <ul key={idx} className="flex flex-col gap-2.5 pl-1.5">
                            {items.map((item, i) => {
                                const content = item.replace(/^•\s*/, '').trim();
                                if (!content) return null;
                                return (
                                    <li key={i} className="flex items-start gap-3">
                                        <span className="text-neutral-300 mt-[7px] text-[10px] shrink-0">■</span>
                                        <span>{content}</span>
                                    </li>
                                );
                            })}
                        </ul>
                    );
                }

                // If section is a header (short, title cased, no punctuation at end, or ends with colon)
                if (
                    section.length < 60 &&
                    (!section.match(/[.!?]$/) || section.endsWith(':')) &&
                    !section.includes('\n')
                ) {
                    return (
                        <h3 key={idx} className="font-semibold text-neutral-900 text-[16px] mt-2 mb-0 tracking-tight">
                            {section.replace(/:$/, '')}
                        </h3>
                    );
                }

                // Regular paragraph
                return (
                    <p key={idx} className="whitespace-pre-wrap">
                        {section}
                    </p>
                );
            })}
        </div>
    );
};

export default function ResponseCard({ query, response, modelUsed, isLoading, isError }: ResponseCardProps) {
    if (!query && !response && !isLoading && !isError) return null;

    return (
        <div className="w-full max-w-2xl mx-auto mt-8 flex flex-col gap-6 animate-[slideUpFade_0.6s_ease-out_forwards]">

            {/* Right-aligned Question Block */}
            {query && (
                <div className="flex justify-end w-full animate-[slideUpFade_0.4s_ease-out_forwards]">
                    <div className="max-w-[85%] bg-neutral-100/60 px-5 py-3 rounded-2xl rounded-tr-md shadow-[0_2px_10px_-4px_rgba(0,0,0,0.02)]">
                        <p className="text-[15px] leading-relaxed text-neutral-800 font-medium text-right">
                            {query}
                        </p>
                    </div>
                </div>
            )}

            {/* Left-aligned Answer Area */}
            {isLoading ? (
                <div className="animate-[slideUpFade_0.5s_ease-out_forwards]">
                    <LoadingCard />
                </div>
            ) : isError ? (
                <div className="flex flex-col w-full bg-[#fdfafb] rounded-3xl p-6 md:p-8 shadow-[0_8px_30px_rgb(0,0,0,0.04)] border border-rose-100/50 animate-[slideUpFade_0.5s_ease-out_forwards]">
                    <div className="flex items-start gap-3 text-rose-600/90 text-[15px] leading-relaxed font-medium">
                        <svg className="w-5 h-5 mt-0.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                        </svg>
                        <p>{response || "An unexpected error occurred."}</p>
                    </div>
                </div>
            ) : response ? (
                <div className="flex flex-col w-full bg-white rounded-3xl p-6 md:p-8 shadow-[0_8px_30px_rgb(0,0,0,0.04)] border border-neutral-100 animate-[slideUpFade_0.5s_ease-out_forwards]">
                    <FormattedResponse text={response} />
                    <MetadataBar modelUsed={modelUsed} />
                </div>
            ) : null}
        </div>
    );
}
