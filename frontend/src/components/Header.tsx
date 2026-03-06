import React from "react";

export default function Header() {
    return (
        <header className="w-full py-12 flex flex-col items-center justify-center fade-in">
            <h1 className="text-3xl md:text-4xl font-semibold tracking-tight text-[#f8fafc] mb-3">
                ⚗️ PharmaGuide
            </h1>
            <p className="text-[14px] md:text-[15px] text-slate-400 font-medium tracking-wide">
                Grounded Drug Information
            </p>
        </header>
    );
}
