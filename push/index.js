const fs = require("node:fs");
const path = require("node:path");
const { spawnSync } = require("node:child_process");

const projectRoot = path.resolve(__dirname, "..");
const argumentsList = process.argv.slice(2);

if (argumentsList.includes("--help") || argumentsList.includes("-h")) {
  console.log("Usage: node push [--amend|--ammend]");
  console.log("       node push");
  console.log("\nNormal mode creates a new commit and pushes normally.");
  console.log("Amend mode updates the latest commit and uses force-with-lease.");
  process.exit(0);
}

const amend = process.argv.slice(2).some(
  (argument) => argument === "--amend" || argument === "--ammend"
);

function run(command, args) {
  console.log(`> ${command} ${args.join(" ")}`);

  const result = spawnSync(command, args, {
    cwd: projectRoot,
    stdio: "inherit",
    shell: false,
  });

  if (result.error) {
    console.error(result.error.message);
    process.exit(1);
  }

  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}

const lessonsPath = path.join(projectRoot, "lessons.txt");
const lessons = fs.readFileSync(lessonsPath, "utf8");
const lessonTitles = lessons.match(/^LESSON \d+ - .+$/gm) ?? [];

if (lessonTitles.length === 0) {
  console.error("No lesson title found in lessons.txt.");
  process.exit(1);
}

const commitMessage = lessonTitles.at(-1).trim();

console.log(`Commit message: ${commitMessage}`);
run("git", ["add", "."]);

if (amend) {
  run("git", ["commit", "--amend", "-m", commitMessage]);
  run("git", ["push", "--force-with-lease", "origin", "main"]);
} else {
  run("git", ["commit", "-m", commitMessage]);
  run("git", ["push", "origin", "main"]);
}
