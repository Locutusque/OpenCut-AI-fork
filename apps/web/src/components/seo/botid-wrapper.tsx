"use client";

import dynamic from "next/dynamic";

const BotIdClient = dynamic(
	() => import("botid/client").then((mod) => mod.BotIdClient),
	{ ssr: false }
);

interface ProtectedRoute {
	path: string;
	method: string;
}

interface BotIdWrapperProps {
	protect: ProtectedRoute[];
}

export default function BotIdWrapper({ protect }: BotIdWrapperProps) {
	return <BotIdClient protect={protect} />;
}
