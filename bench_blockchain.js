const fs = require("fs");
const crypto = require("crypto");
const { ethers } = require("ethers");

const cfg = JSON.parse(fs.readFileSync(process.argv[2] || "bench_config.json", "utf8"));
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
const p95 = (a) => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length * 0.95)]; };
let nextId = Date.now() % 1e9;

function fill(args, accounts, id) {
  return args.map((a) => {
    if (a === "$i") return id;
    if (a === "$cid") return "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi";
    if (a === "$sha256") return "0x" + crypto.createHash("sha256").update(String(id)).digest("hex");
    if (a === "$sha3") return "0x" + crypto.createHash("sha3-256").update(String(id)).digest("hex");
    if (typeof a === "string" && a.startsWith("$account")) return accounts[Number(a.slice(8))];
    return a;
  });
}

async function timedTx(contract, fn, args) {
  const t0 = performance.now();
  const tx = await contract[fn](...args);
  const rc = await tx.wait();
  return { gas: Number(rc.gasUsed), ms: performance.now() - t0 };
}

async function main() {
  const provider = new ethers.JsonRpcProvider(cfg.rpc);
  const accounts = (await provider.send("eth_accounts", [])).map(ethers.getAddress);
  const signer = await provider.getSigner(accounts[0]);
  const abi = JSON.parse(fs.readFileSync(cfg.abi, "utf8"));
  const contract = new ethers.Contract(cfg.address, abi, signer);
  const out = { functions: [], throughput: [], ipfs: [] };

  const reg = { gas: [], ms: [] };
  const per = Object.fromEntries(cfg.calls.map((c) => [c.fn, { gas: [], ms: [] }]));
  for (let r = 0; r < cfg.repeats; r++) {
    const id = nextId++;
    const g = await timedTx(contract, cfg.registerFn, fill(cfg.registerArgs, accounts, id));
    reg.gas.push(g.gas); reg.ms.push(g.ms);
    for (const c of cfg.calls) {
      const res = await timedTx(contract, c.fn, fill(c.args, accounts, id));
      per[c.fn].gas.push(res.gas); per[c.fn].ms.push(res.ms);
    }
  }
  for (const [fn, v] of [[cfg.registerFn, reg], ...Object.entries(per)]) {
    out.functions.push({ function: fn, gasMedian: median(v.gas), latencyMsMedian: +median(v.ms).toFixed(1),
                         latencyMsP95: +p95(v.ms).toFixed(1) });
  }

  for (const n of cfg.batchSizes) {
    const signers = await Promise.all(accounts.slice(0, Math.min(10, accounts.length)).map((a) => provider.getSigner(a)));
    const nonces = await Promise.all(signers.map((s) => provider.getTransactionCount(s.address, "pending")));
    const t0 = performance.now();
    const txs = [];
    for (let k = 0; k < n; k++) {
      const s = k % signers.length;
      const c = contract.connect(signers[s]);
      txs.push(c[cfg.registerFn](...fill(cfg.registerArgs, accounts, nextId++), { nonce: nonces[s]++ }));
    }
    const sent = await Promise.all(txs);
    await Promise.all(sent.map((t) => t.wait()));
    const sec = (performance.now() - t0) / 1000;
    out.throughput.push({ batch: n, seconds: +sec.toFixed(2), tps: +(n / sec).toFixed(2) });
  }

  for (const kb of cfg.ipfsSizesKB || []) {
    const add = [], cat = [];
    for (let r = 0; r < 5; r++) {
      const body = new FormData();
      body.append("file", new Blob([crypto.randomBytes(kb * 1024)]), "record.bin");
      let t0 = performance.now();
      const res = await (await fetch(`${cfg.ipfsApi}/api/v0/add?pin=true`, { method: "POST", body })).json();
      add.push(performance.now() - t0);
      t0 = performance.now();
      await (await fetch(`${cfg.ipfsApi}/api/v0/cat?arg=${res.Hash}`, { method: "POST" })).arrayBuffer();
      cat.push(performance.now() - t0);
    }
    out.ipfs.push({ sizeKB: kb, addMsMedian: +median(add).toFixed(1), catMsMedian: +median(cat).toFixed(1) });
  }

  fs.writeFileSync("blockchain_results.json", JSON.stringify(out, null, 2));
  console.table(out.functions); console.table(out.throughput); console.table(out.ipfs);
}

main().catch((e) => { console.error(e); process.exit(1); });
