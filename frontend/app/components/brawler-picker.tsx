import BrawlerIcon from "./brawler-icon";
import brawlerJson from "../../../backend/src/out/brawlers/brawlers.json"

function get_brawlers(): string[] {
  const brawlers: string[] = Object.keys(brawlerJson);
  return brawlers;
}

export default function BrawlerPicker() {
    const brawlers:string[] = get_brawlers()
    return (
      <div className="grid grid-cols-3 gap-2 p-4">
        {brawlers.map((brawler) => (
          <BrawlerIcon key={brawler} brawler={brawler} />
        ))}
      </div>
    );
  }