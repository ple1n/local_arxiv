declare module "latex.js" {
  export class HtmlGenerator {
    constructor(options?: { hyphenate?: boolean });
  }

  export function parse(
    source: string,
    options?: { generator?: HtmlGenerator },
  ): { htmlDocument(): Document };
}

declare module "*.css?inline" {
  const css: string;
  export default css;
}
