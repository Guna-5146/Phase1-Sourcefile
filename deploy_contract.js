const fs = require("fs");
const solc = require("solc");
const { ethers } = require("ethers");

async function main() {
  const file = process.argv[2] || "MaduraiLandRegistry.sol";
  const rpc = process.argv[3] || "http://127.0.0.1:8545";
  const input = {
    language: "Solidity",
    sources: { [file]: { content: fs.readFileSync(file, "utf8") } },
    settings: { optimizer: { enabled: true, runs: 200 },
                outputSelection: { "*": { "*": ["abi", "evm.bytecode.object"] } } },
  };
  const out = JSON.parse(solc.compile(JSON.stringify(input)));
  const errors = (out.errors || []).filter((e) => e.severity === "error");
  if (errors.length) throw new Error(errors.map((e) => e.formattedMessage).join("\n"));
  const [name, c] = Object.entries(out.contracts[file])[0];
  fs.mkdirSync("build", { recursive: true });
  fs.writeFileSync(`build/${name}.abi.json`, JSON.stringify(c.abi, null, 2));
  console.log(`Compiled ${name} with solc ${solc.version()}`);

  const provider = new ethers.JsonRpcProvider(rpc);
  const accounts = (await provider.send("eth_accounts", [])).map(ethers.getAddress);
  const admin = await provider.getSigner(accounts[0]);
  const contract = await new ethers.ContractFactory(c.abi, c.evm.bytecode.object, admin).deploy();
  const rc = await contract.deploymentTransaction().wait();
  const address = await contract.getAddress();
  console.log(`Deployed at ${address} (deployment gas ${rc.gasUsed})`);
  for (const a of accounts.slice(1, 10)) await (await contract.setRegistrar(a, true)).wait();

  const cfg = {
    rpc, ipfsApi: "http://127.0.0.1:5001", abi: `./build/${name}.abi.json`, address,
    repeats: 20, batchSizes: [10, 50, 100, 200], registerFn: "registerLand",
    registerArgs: ["$i", "$account1", "$cid", "$sha256", "$sha3", 125, 4500000],
    calls: [
      { fn: "transferOwnership", args: ["$i", "$account2"] },
      { fn: "updateLandPrice", args: ["$i", 5000000] },
      { fn: "updateLandArea", args: ["$i", 130] },
      { fn: "verifyLand", args: ["$i"] },
    ],
    ipfsSizesKB: [1, 10, 100, 1000, 5000],
  };
  fs.writeFileSync("bench_config.json", JSON.stringify(cfg, null, 2));
  console.log("Wrote bench_config.json");
}

main().catch((e) => { console.error(e); process.exit(1); });
