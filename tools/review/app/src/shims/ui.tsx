/**
 * TEMPORARY SHIM for `ShortcutList` from `@vitavision/ui`, contributed upstream on lab-ui branch
 * `feat/ui-shortcut-list` and not released yet. Same props and markup as the upstream
 * component; once it is released, delete this file and import `ShortcutList`,
 * `ShortcutGroup` from "@vitavision/ui" instead.
 */

import { Kbd, cn } from "@vitavision/ui";
import { Fragment, useId, type ComponentProps } from "react";

/** One command and the keys that run it. */
export interface Shortcut {
  keys: readonly string[];
  description: string;
}

/** A titled group of shortcuts. */
export interface ShortcutGroup {
  title: string;
  shortcuts: readonly Shortcut[];
}

export type ShortcutListProps = Omit<ComponentProps<"div">, "children"> & { groups: readonly ShortcutGroup[] };

/** A screen's keyboard shortcuts as titled groups (see the upstream component). */
export function ShortcutList({ groups, className, ...rest }: ShortcutListProps) {
  const baseId = useId();
  return (
    <div {...rest} className={cn("grid gap-x-8 gap-y-4 sm:grid-cols-2", className)}>
      {groups.map((group, g) =>
        group.shortcuts.length === 0 ? null : (
          <section key={group.title} aria-labelledby={`${baseId}-${g}`} className="min-w-0">
            <h3 id={`${baseId}-${g}`} className="mb-1.5 text-[11px] font-semibold tracking-wider text-fg-muted uppercase">
              {group.title}
            </h3>
            <dl className="grid grid-cols-[auto_1fr] items-baseline gap-x-3 gap-y-1 text-sm">
              {group.shortcuts.map((shortcut) => (
                <Fragment key={shortcut.description}>
                  <dt className="flex flex-wrap items-center gap-1 text-xs text-fg-muted">
                    {shortcut.keys.map((key, k) => (
                      <Fragment key={key}>
                        {k > 0 && <span>or</span>}
                        <Kbd>{key}</Kbd>
                      </Fragment>
                    ))}
                  </dt>
                  <dd className="text-fg">{shortcut.description}</dd>
                </Fragment>
              ))}
            </dl>
          </section>
        ),
      )}
    </div>
  );
}
