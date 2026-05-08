/**
 * INDILA · Shield mark
 *
 * A brand-faithful reconstruction of the INDILA shield with torch, adapted
 * from the live site's logo SVG. Uses the exact brand palette:
 *   - Shield: #15355B (navy)
 *   - Torch:  #D4A029 (gold)
 *   - Stars:  #DBA428 (gold stroke variant)
 *
 * The original live-site logo is composed of the same elements (shield silhouette,
 * central torch, flanking star column, curved flourish beneath). Proportions
 * and palette match; exact path coordinates are redrawn so this file is
 * self-contained and does not require shipping the original asset.
 *
 * TODO(brand): once INDILA provides their official SVG logo file, swap the
 * paths in this component out for the canonical asset.
 */

export type IndilaMarkProps = {
  /** Rendered height in px. Width scales proportionally. */
  size?: number;
  /** Shield body color. Defaults to INDILA navy. */
  navy?: string;
  /** Torch / ornament color. Defaults to INDILA gold. */
  gold?: string;
  /** Decorative stroke color (curved flourish). Defaults to gold stroke variant. */
  goldStroke?: string;
  /** Optional accessible label. */
  title?: string;
};

export default function IndilaMark({
  size = 48,
  navy = '#15355B',
  gold = '#D4A029',
  goldStroke = '#DBA428',
  title = 'INDILA',
}: IndilaMarkProps) {
  // viewBox is designed around a 72×82 unit composition.
  const w = (size * 72) / 82;
  return (
    <svg
      role="img"
      aria-label={title}
      width={w}
      height={size}
      viewBox="0 0 72 82"
      xmlns="http://www.w3.org/2000/svg"
      style={{ display: 'block' }}
    >
      <title>{title}</title>

      {/* Shield silhouette. Three concentric contours like the live logo. */}
      <path
        d="M36 6 H62 V37 C62 57 47 68 36 72 C25 68 10 57 10 37 V6 Z"
        fill={navy}
        stroke={navy}
        strokeWidth="1"
        strokeLinejoin="round"
      />
      <path
        d="M36 10 H58 V36 C58 54 45 64 36 67 C27 64 14 54 14 36 V10 Z"
        fill="none"
        stroke={goldStroke}
        strokeWidth="0.8"
        strokeLinejoin="round"
        opacity="0.5"
      />

      {/* Torch flame */}
      <path
        d="M31.5 26
           C28.5 23.5 29.5 19.5 32 18
           C32.5 19 33.5 19.5 34 20
           C34 16.5 37 14 40.5 14.5
           C39.5 16 39 17.5 40 19
           C41.5 22 42.5 25 38 27.5
           C34 29.5 31.5 26 31.5 26 Z"
        fill={gold}
      />

      {/* Torch cup + handle */}
      <rect x="32" y="29" width="8" height="1.6" fill={gold} />
      <path d="M32.5 30.6 h7 c-2 2 -1.7 4.5 -1.7 6.3 h-3.6 c0 -1.8 0.2 -4.3 -1.7 -6.3 Z" fill={gold} />
      <rect x="34" y="37" width="4" height="14" fill={gold} />
      <rect x="33" y="51.5" width="6" height="1.4" fill={gold} />

      {/* Curved flourish beneath */}
      <path
        d="M16 45 C18 56 26 65 36 70 C46 65 54 56 56 45"
        fill="none"
        stroke={goldStroke}
        strokeWidth="1.4"
        strokeLinecap="round"
      />

      {/* Flanking stars. Three each side, like the live logo. */}
      {[20, 29, 38].map((cy) => (
        <g key={'l' + cy} transform={`translate(7,${cy})`}>
          <Star size={2} fill={goldStroke} />
        </g>
      ))}
      {[20, 29, 38].map((cy) => (
        <g key={'r' + cy} transform={`translate(65,${cy})`}>
          <Star size={2} fill={goldStroke} />
        </g>
      ))}
    </svg>
  );
}

function Star({ size = 2, fill = '#DBA428' }: { size?: number; fill?: string }) {
  // 5-point star centered at (0,0)
  const s = size;
  const d =
    `M0 ${-s} L${s * 0.3} ${-s * 0.3} L${s} ${-s * 0.1} L${s * 0.4} ${s * 0.35} ` +
    `L${s * 0.6} ${s} L0 ${s * 0.55} L${-s * 0.6} ${s} L${-s * 0.4} ${s * 0.35} ` +
    `L${-s} ${-s * 0.1} L${-s * 0.3} ${-s * 0.3} Z`;
  return <path d={d} fill={fill} />;
}

/**
 * Full horizontal lockup. Mark + INDILA wordmark + tagline.
 * Mirrors the treatment of the live site header.
 */
export function IndilaLogo({
  onDark = false,
  showTagline = true,
  size = 36,
}: {
  onDark?: boolean;
  showTagline?: boolean;
  size?: number;
}) {
  const navy = onDark ? '#FAF7F0' : '#15355B';
  const taglineColor = onDark ? 'rgba(250,247,240,0.55)' : '#7C837A';
  return (
    <div style={{ display: 'inline-flex', alignItems: 'center', gap: '12px', lineHeight: 1 }}>
      <IndilaMark
        size={size}
        navy={onDark ? '#FAF7F0' : '#15355B'}
        gold={onDark ? '#D4A029' : '#D4A029'}
        goldStroke={onDark ? '#DBA428' : '#DBA428'}
      />
      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
        <span
          className="wordmark"
          style={{
            fontSize: size * 0.40,
            lineHeight: 1,
            color: navy,
          }}
        >
          INDILA
        </span>
        {showTagline && (
          <span
            style={{
              fontSize: Math.max(8, size * 0.18),
              letterSpacing: '0.14em',
              textTransform: 'uppercase',
              color: taglineColor,
              fontWeight: 500,
              lineHeight: 1,
            }}
          >
            Indiana Individualized Learning Academies
          </span>
        )}
      </div>
    </div>
  );
}
