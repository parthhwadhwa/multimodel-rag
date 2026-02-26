import React from "react";

export default function LoadingCard() {
    return (
        <div className="flex flex-col w-full bg-white rounded-3xl p-6 md:p-8 shadow-[0_8px_30px_rgb(0,0,0,0.04)] border border-neutral-100 animate-[pulse_2s_cubic-bezier(0.4,0,0.6,1)_infinite]">
            <div className="flex flex-col gap-4 w-full max-w-[70ch]">
                {/* Simulated Header */}
                <div className="h-5 bg-neutral-100/80 rounded-md w-1/3 mb-2"></div>

                {/* Simulated Paragraph */}
                <div className="space-y-3">
                    <div className="h-4 bg-neutral-100 rounded-md w-[90%]"></div>
                    <div className="h-4 bg-neutral-100 rounded-md w-full"></div>
                    <div className="h-4 bg-neutral-100 rounded-md w-[80%]"></div>
                </div>

                {/* Simulated Bullets */}
                <div className="space-y-3 mt-2 pl-4">
                    <div className="flex items-center gap-3">
                        <div className="w-1.5 h-1.5 rounded-full bg-neutral-200"></div>
                        <div className="h-4 bg-neutral-100 rounded-md w-[60%]"></div>
                    </div>
                    <div className="flex items-center gap-3">
                        <div className="w-1.5 h-1.5 rounded-full bg-neutral-200"></div>
                        <div className="h-4 bg-neutral-100 rounded-md w-[75%]"></div>
                    </div>
                </div>
            </div>
        </div>
    );
}
