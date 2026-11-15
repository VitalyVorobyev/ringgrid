/**
 * TEMPORARY SHIM for `ImageLayer`'s `gain` and `gamma` props, contributed upstream to
 * `@vitavision/stage2d` on lab-ui branch `feat/stage2d-image-adjust` and not released yet.
 * Same props and the same tone curve (each channel `v` drawn as `gain × v^(1 / gamma)`, sRGB,
 * through one SVG filter); once released, delete this file and import `ImageLayer` from
 * "@vitavision/stage2d" instead.
 *
 * The released layer takes no inline style, so the filter goes on a full-size wrapper here
 * rather than on the images themselves; what is drawn is the same.
 */

import { ImageLayer as BaseImageLayer, type ImageLayerProps as BaseProps } from "@vitavision/stage2d";
import { useId } from "react";

export type ImageLayerProps = BaseProps & {
  /** Display gain, after `gamma`. Defaults to 1; negative counts as 0, non-finite as 1. */
  gain?: number | undefined;
  /** Display gamma: `gain × v^(1 / gamma)`. Defaults to 1; a value that is not positive counts as 1. */
  gamma?: number | undefined;
};

function toneCurve(gain: number | undefined, gamma: number | undefined): { amplitude: number; exponent: number } | null {
  const amplitude = gain === undefined || !Number.isFinite(gain) ? 1 : Math.max(0, gain);
  const g = gamma === undefined || !(gamma > 0) || !Number.isFinite(gamma) ? 1 : gamma;
  if (amplitude === 1 && g === 1) return null;
  return { amplitude, exponent: 1 / g };
}

/** `ImageLayer` with a display tone curve. */
export function ImageLayer({ gain, gamma, ...props }: ImageLayerProps) {
  const curve = toneCurve(gain, gamma);
  const filterId = `stage-image-tone-${useId().replace(/[^A-Za-z0-9_-]/g, "")}`;
  if (!curve) return <BaseImageLayer {...props} />;
  return (
    <div className="pointer-events-none absolute inset-0" style={{ filter: `url(#${filterId})` }} data-adjusted="">
      <svg aria-hidden="true" width="0" height="0" className="absolute" focusable="false">
        <filter id={filterId} colorInterpolationFilters="sRGB" data-tone-curve="">
          <feComponentTransfer>
            <feFuncR type="gamma" amplitude={curve.amplitude} exponent={curve.exponent} offset={0} />
            <feFuncG type="gamma" amplitude={curve.amplitude} exponent={curve.exponent} offset={0} />
            <feFuncB type="gamma" amplitude={curve.amplitude} exponent={curve.exponent} offset={0} />
          </feComponentTransfer>
        </filter>
      </svg>
      <BaseImageLayer {...props} />
    </div>
  );
}
