import React from "react";

export default function Header() {
    return (
        <header
            style={{
                width: "100%",
                borderBottom: "1px solid var(--border)",
                background: "var(--header-bg)",
                padding: "16px 24px",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
            }}
        >
            <h1
                style={{
                    fontSize: "20px",
                    fontWeight: 600,
                    color: "var(--text-primary)",
                    margin: 0,
                }}
            >
                💊 PharmaGuide
            </h1>
            <p
                style={{
                    fontSize: "13px",
                    color: "var(--text-secondary)",
                    marginTop: "4px",
                }}
            >
                AI-powered Drug Information Assistant
            </p>
        </header>
    );
}
