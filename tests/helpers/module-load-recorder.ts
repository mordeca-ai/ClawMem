/**
 * Bun preload: appends the path of every src/*.ts module the process loads to the file named by
 * CLAWMEM_LOAD_RECORD. Used by tests/unit/cli-startup-import-graph.test.ts (vn4rz.116) to assert
 * which modules a CLI verb pulls in. The loader passes sources through unchanged.
 */
import { plugin } from "bun";
import { appendFileSync, readFileSync } from "fs";

const record = process.env.CLAWMEM_LOAD_RECORD;
if (record) {
  plugin({
    name: "clawmem-load-recorder",
    setup(build) {
      build.onLoad({ filter: /[\\/]src[\\/].*\.ts$/ }, (args) => {
        appendFileSync(record, args.path + "\n");
        return { contents: readFileSync(args.path, "utf8"), loader: "ts" };
      });
    },
  });
}
