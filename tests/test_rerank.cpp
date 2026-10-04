// Reranking on the wire, without an Engine: the Jina request and its TEI variant, the judging
// prompt each document becomes, the score read from the first answer token's alternatives, and the
// ranked response.

#include "serve/generation_service.h"
#include "serve/rerank.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer::serve;
using Json = nlohmann::json;

int failures = 0;

void expect(bool condition, const std::string& label) {
    if (condition) { return; }
    std::cerr << "expectation failed: " << label << '\n';
    ++failures;
}

std::string refusal(const char* body) {
    try {
        (void)parse_rerank_request(RequestJson::parse(body));
    } catch (const ApiException& error) { return error.error().param; }
    return {};
}

GenerationOutcome judged(std::vector<std::pair<std::string, double>> alternatives) {
    GenerationOutcome out;
    FirstTokenLogprobsView logprobs;
    for (const auto& [bytes, probability] : alternatives) {
        logprobs.top.push_back(
            {.bytes = bytes, .logprob = static_cast<float>(std::log(probability))});
    }
    logprobs.selected        = logprobs.top.front();
    out.first_token_logprobs = std::move(logprobs);
    return out;
}

bool near(double a, double b) { return std::abs(a - b) < 1e-5; }

void parsing() {
    const auto jina = parse_rerank_request(RequestJson::parse(
        R"({"model": "m", "query": "q", "documents": ["a", {"text": "b"}, "c"], "top_n": 9})"));
    expect(jina.documents == std::vector<std::string>({"a", "b", "c"}) && !jina.tei,
           "documents as strings and text objects");
    expect(jina.top_n == 3 && jina.return_documents && jina.model == "m",
           "top_n is clamped to the documents and documents are returned by default");
    expect(!jina.instruction.empty(), "a default instruction");
    const auto tei = parse_rerank_request(
        RequestJson::parse(R"({"query": "q", "texts": ["a"], "return_text": true})"));
    expect(tei.tei && tei.return_documents, "TEI's texts and return_text");
    expect(refusal(R"({"documents": ["a"]})") == "query", "a query is required");
    expect(refusal(R"({"query": "q", "documents": []})") == "documents",
           "at least one document is required");
    expect(refusal(R"({"query": "q", "documents": ["a"], "top_n": 0})") == "top_n",
           "top_n must be positive");
    expect(refusal(R"({"query": "q", "documents": [3]})") == "documents", "a document is text");

    const auto judgement = make_rerank_judgement(jina, "b");
    expect(judgement.messages.size() == 2 &&
               judgement.messages[0].role == ninfer::ChatRole::System &&
               judgement.messages[1].content.at(0).text.find("<Query>: q\n<Document>: b") !=
                   std::string::npos,
           "the judging prompt holds the query and the document");
    expect(judgement.enable_thinking == false && judgement.max_tokens == 1 &&
               judgement.first_token_top_logprobs == ninfer::kMaximumFirstTokenTopLogprobs &&
               judgement.graft && judgement.graft->empty(),
           "one answer token with its alternatives, no thinking, no graft");
}

void scoring() {
    expect(near(rerank_score(judged({{"yes", 0.6}, {"no", 0.2}, {"maybe", 0.1}})), 0.75),
           "P(yes) / (P(yes) + P(no))");
    expect(near(rerank_score(judged({{"no", 0.5}, {"Yes", 0.2}, {"yes", 0.1}, {"No", 0.1}})),
                0.3 / 0.9),
           "capitalized answers count with their lower-case forms");
    expect(near(rerank_score(judged({{"yes", 0.9}, {"the", 0.01}})), 0.9 / 0.91),
           "a missing answer counts as likely as the least likely alternative");
    expect(near(rerank_score(judged({{"the", 0.5}, {"a", 0.2}})), 0.5),
           "a judgement that answers neither scores one half");
}

void response() {
    const auto request = parse_rerank_request(RequestJson::parse(
        R"({"model": "m", "query": "q", "documents": ["a", "b", "c"], "top_n": 2})"));
    const Json body    = Json::parse(make_rerank_response(
        request,
        {{.index = 0, .score = 0.2}, {.index = 1, .score = 0.9}, {.index = 2, .score = 0.9}}, 30));
    expect(body["object"] == "list" && body["model"] == "m" && body["usage"]["prompt_tokens"] == 30,
           "a Jina list with usage");
    expect(body["results"].size() == 2 && body["results"][0]["index"] == 1 &&
               body["results"][1]["index"] == 2 && body["results"][0]["document"]["text"] == "b",
           "best first, the lower index first among equals, top_n kept, documents returned");
    const auto tei =
        parse_rerank_request(RequestJson::parse(R"({"query": "q", "texts": ["a", "b"]})"));
    const Json array = Json::parse(
        make_rerank_response(tei, {{.index = 0, .score = 0.1}, {.index = 1, .score = 0.8}}, 10));
    expect(array.is_array() && array[0]["index"] == 1 && array[0]["score"] == 0.8 &&
               !array[0].contains("text"),
           "TEI's bare array without texts unless asked");
}

} // namespace

int main() {
    try {
        parsing();
        scoring();
        response();
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " expectation(s) failed\n";
        return 1;
    }
    std::cout << "rerank: ok\n";
    return 0;
}
