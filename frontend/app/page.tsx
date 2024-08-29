"use client";
import Header from "./components/header"
import { BrawlerProvider } from './components/brawler-context';
import Picking from "./components/picking";
import Selection from "./components/selection";


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
