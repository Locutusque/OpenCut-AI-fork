import { Ratelimit } from "@upstash/ratelimit";
import { Redis } from "@upstash/redis";

let baseRateLimit: Ratelimit | null = null;

function getBaseRateLimit() {
	if (!baseRateLimit) {
		const redis = new Redis({
			url: process.env.UPSTASH_REDIS_REST_URL || "http://localhost:8079",
			token: process.env.UPSTASH_REDIS_REST_TOKEN || "example_token",
		});

		baseRateLimit = new Ratelimit({
			redis,
			limiter: Ratelimit.slidingWindow(100, "1 m"), // 100 requests per minute
			analytics: true,
			prefix: "rate-limit",
		});
	}

	return baseRateLimit;
}

export async function checkRateLimit({ request }: { request: Request }) {
	const ip = request.headers.get("x-forwarded-for") ?? "anonymous";
	const { success } = await getBaseRateLimit().limit(ip);
	return { success, limited: !success };
}
