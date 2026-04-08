"use client";
import dynamic from "next/dynamic";
import Header from "./components/header"
import { BrawlerProvider } from './components/brawler-context';

const Picking = dynamic(() => import("./components/picking"), { ssr: false });
const Selection = dynamic(() => import("./components/selection"), { ssr: false });


export default function Home() {
  return (
    <BrawlerProvider>
      <main className="mx-auto md:px-[30px] px-[10px]">
        <Header/>
        <Picking/>
        <Selection/>
      </main>
    </BrawlerProvider>
  );
}
