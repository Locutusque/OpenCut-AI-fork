import { useCallback, useEffect, useRef, useState } from "react";
import { useEditor } from "@/hooks/use-editor";
import type { MediaAsset } from "@/types/assets";
import type { VideoElement, ImageElement } from "@/types/timeline";

interface FilmstripState {
	thumbnails: string[];
	loading: boolean;
}

const CACHE = new Map<string, string[]>();
const MAX_CACHE_SIZE = 100;

function pruneCache() {
	if (CACHE.size > MAX_CACHE_SIZE) {
		const keys = Array.from(CACHE.keys());
		for (let i = 0; i < keys.length - MAX_CACHE_SIZE; i++) {
			CACHE.delete(keys[i]);
		}
	}
}

async function generateFilmstrip({
	mediaAsset,
	numFrames,
	width,
	height,
	sourceStart,
	sourceEnd,
}: {
	mediaAsset: MediaAsset;
	numFrames: number;
	width: number;
	height: number;
	sourceStart: number;
	sourceEnd: number;
}): Promise<string[]> {
	if (mediaAsset.type !== "video" || !mediaAsset.url) return [];

	const cacheKey = `${mediaAsset.id}-${numFrames}-${width}-${sourceStart.toFixed(3)}-${sourceEnd.toFixed(3)}`;
	const cached = CACHE.get(cacheKey);
	if (cached) return cached;

	return new Promise((resolve) => {
		const video = document.createElement("video");
		video.crossOrigin = "anonymous";
		video.muted = true;
		video.preload = "auto";

		const timeout = setTimeout(() => {
			video.src = "";
			resolve([]);
		}, 10000);

		video.onloadedmetadata = () => {
			const duration = video.duration;
			if (!duration || !isFinite(duration)) {
				clearTimeout(timeout);
				resolve([]);
				return;
			}

			// Sample within the clip's trimmed source range, clamped to the
			// actual media duration.
			const rangeStart = Math.max(0, Math.min(sourceStart, duration));
			const rangeEnd = Math.max(rangeStart, Math.min(sourceEnd, duration));
			const span = rangeEnd - rangeStart;

			const frames: string[] = [];
			const interval = span / numFrames;

			const captureFrame = (index: number) => {
				if (index >= numFrames) {
					clearTimeout(timeout);
					CACHE.set(cacheKey, frames);
					pruneCache();
					resolve(frames);
					return;
				}

				video.currentTime = rangeStart + interval * (index + 0.5);
			};

			video.onseeked = () => {
				const canvas = document.createElement("canvas");
				canvas.width = width;
				canvas.height = height;
				const ctx = canvas.getContext("2d");
				if (!ctx) {
					clearTimeout(timeout);
					resolve(frames);
					return;
				}
				ctx.drawImage(video, 0, 0, width, height);
				frames.push(canvas.toDataURL("image/jpeg", 0.4));
				captureFrame(frames.length);
			};

			captureFrame(0);
		};

		video.onerror = () => {
			clearTimeout(timeout);
			resolve([]);
		};

		video.src = mediaAsset.url ?? "";
	});
}

export function useFilmstrip({
	mediaAsset,
	clipDuration,
	visibleWidth,
	trackHeight,
	trimStart = 0,
	playbackRate = 1,
}: {
	mediaAsset: MediaAsset | null;
	clipDuration: number;
	visibleWidth: number;
	trackHeight: number;
	trimStart?: number;
	playbackRate?: number;
}): FilmstripState {
	const [state, setState] = useState<FilmstripState>({ thumbnails: [], loading: false });
	const abortRef = useRef(false);

	const generate = useCallback(async () => {
		if (!mediaAsset || mediaAsset.type !== "video" || clipDuration <= 0) {
			setState({ thumbnails: [], loading: false });
			return;
		}

		const thumbWidth = Math.max(40, Math.round(trackHeight * 16 / 9));
		const numFrames = Math.max(1, Math.min(20, Math.ceil(visibleWidth / thumbWidth)));

		// Map the timeline clip back to its source range so each thumbnail
		// reflects the portion of the video that actually plays.
		const sourceStart = trimStart;
		const sourceEnd = trimStart + clipDuration * playbackRate;

		setState({ thumbnails: [], loading: true });

		const thumbnails = await generateFilmstrip({
			mediaAsset,
			numFrames,
			width: thumbWidth,
			height: trackHeight,
			sourceStart,
			sourceEnd,
		});

		if (!abortRef.current) {
			setState({ thumbnails, loading: false });
		}
	}, [mediaAsset?.id, clipDuration, visibleWidth, trackHeight, trimStart, playbackRate]);

	useEffect(() => {
		abortRef.current = false;
		generate();
		return () => {
			abortRef.current = true;
		};
	}, [generate]);

	return state;
}
