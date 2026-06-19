import Link from "next/link";
import { Header } from "@/components/header";
import { Footer } from "@/components/footer";

export const dynamic = "force-dynamic";

export default function NotFound() {
	return (
		<div className="flex min-h-screen flex-col">
			<Header />
			<main className="flex flex-1 flex-col items-center justify-center bg-background px-4 text-center">
				<div className="space-y-6">
					<h1 className="text-9xl font-extrabold tracking-tight text-muted-foreground/30">
						404
					</h1>
					<h2 className="text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
						Page not found
					</h2>
					<p className="mx-auto max-w-md text-muted-foreground">
						Sorry, we couldn&apos;t find the page you&apos;re looking for. It might have been moved or deleted.
					</p>
					<div className="flex justify-center">
						<Link
							href="/"
							className="inline-flex h-10 items-center justify-center rounded-md bg-primary px-8 text-sm font-medium text-primary-foreground shadow transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
						>
							Go back home
						</Link>
					</div>
				</div>
			</main>
			<Footer />
		</div>
	);
}
