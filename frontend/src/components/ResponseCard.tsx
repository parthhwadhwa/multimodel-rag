import React from "react";
import LoadingCard from "./LoadingCard";
import MetadataBar from "./MetadataBar";

interface ResponseCardProps {
    query: string;
    response: string;
    modelUsed?: string;
    citations?: any[];
    confidenceScore?: number | null;
    isLoading?: boolean;
    isError?: boolean;
}

// Simple text formatter to handle basic markdown-like structures
const FormattedResponse = ({ text }: { text: string }) => {
    // Basic markdown list normalization: handles '1. ', '- ', '* '
    const normalizedText = text.replace(/^(?:\s*)(?:[-*]|\d+\.)\s+(.+)$/gm, '• $1');
    const sections = normalizedText.split('\n\n').filter(Boolean);

    return (
        <div className="flex flex-col gap-5 text-[15px] leading-relaxed text-slate-300 max-w-[70ch]">
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
                                        <span className="text-slate-500 mt-[7px] text-[10px] shrink-0">■</span>
                                        <span className="text-[#f8fafc]">{content}</span>
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
                        <h3 key={idx} className="font-semibold text-[#f8fafc] text-[16px] mt-2 mb-0 tracking-tight">
                            {section.replace(/:$/, '')}
                        </h3>
                    );
                }

                // Regular paragraph
                return (
                    <p key={idx} className="whitespace-pre-wrap text-slate-300">
                        {section}
                    </p>
                );
            })}
        </div>
    );
};

export default function ResponseCard({ query, response, citations, confidenceScore, modelUsed, isLoading, isError }: ResponseCardProps) {
    if (!query && !response && !isLoading && !isError) return null;

    return (
        <div className="w-full max-w-2xl mx-auto mt-8 flex flex-col gap-6 animate-[slideUpFade_0.6s_ease-out_forwards]">

            {/* Right-aligned Question Block */}
            {query && (
                <div className="flex justify-end w-full animate-[slideUpFade_0.4s_ease-out_forwards]">
                    <div className="max-w-[75%] bg-[#1e293b] border border-[#334155]/60 px-5 py-3.5 rounded-3xl rounded-tr-md shadow-sm">
                        <p className="text-[15px] leading-relaxed text-[#f8fafc] font-medium text-left">
                            {query}
                        </p>
                    </div>
                </div>
            )}

            {/* Left-aligned Answer Area */}
            {isLoading ? (
                <div className="flex justify-start w-full animate-[slideUpFade_0.5s_ease-out_forwards]">
                    <div className="flex items-center gap-3 text-slate-400 font-medium bg-[#1e293b] border border-[#334155] px-5 py-4 rounded-3xl rounded-tl-md shadow-sm">
                        <svg className="animate-spin w-5 h-5 text-slate-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Generating response...
                    </div>
                </div>
            ) : isError ? (
                <div className="flex flex-col w-full bg-[#1e293b]/80 rounded-3xl p-6 md:p-8 shadow-sm border border-rose-900/50 animate-[slideUpFade_0.5s_ease-out_forwards]">
                    <div className="flex items-start gap-3 text-rose-400 text-[15px] leading-relaxed font-medium">
                        <svg className="w-5 h-5 mt-0.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                        </svg>
                        <p>{response || "An unexpected error occurred."}</p>
                    </div>
                </div>
            ) : response ? (
                <div className="flex flex-col w-full bg-[#1e293b] rounded-3xl rounded-tl-md p-6 md:p-8 shadow-[0_8px_30px_rgb(0,0,0,0.12)] border border-[#334155] animate-[slideUpFade_0.5s_ease-out_forwards]">
                    <h2 className="text-[20px] font-bold text-[#f8fafc] mb-6 tracking-tight">## Answer</h2>

                    <div className="mb-8">
                        <FormattedResponse text={response} />
                    </div>

                    <div className="pt-6 border-t border-[#334155]/60">
                        {confidenceScore !== undefined && confidenceScore !== null && (
                            <p className="text-[15px] text-slate-300 font-medium mb-4">
                                Confidence: {confidenceScore}%
                            </p>
                        )}

                        {citations && citations.length > 0 && (
                            <div className="text-[14px] text-slate-400">
                                <strong className="block mb-3 text-slate-300 text-[15px]">Sources</strong>
                                <ul className="list-disc pl-5 space-y-1.5 ml-1">
                                    {citations.map((c, i) => (
                                        <li key={i} className="leading-snug">
                                            {c.document} (page {c.page})
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        )}
                    </div>

                    <div className="mt-6">
                        <MetadataBar modelUsed={modelUsed} />
                    </div>
                </div>
            ) : null}
        </div>
    );
}
