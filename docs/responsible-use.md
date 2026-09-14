# Responsible Use of KittyLM

KittyLM is a personal research project. These expectations apply to any output it produces,
now and in future releases.

1. **It is a small research model.** Treat outputs as experimental. Performance on the
   training corpus is not general intelligence.
2. **Validate generated code before using it.** Run tests, review it, and check it for
   security problems. Never execute generated code you have not read on a system you care
   about.
3. **Cybersecurity content is educational and defensive research assistance only.** It is not
   authorization to test systems you do not own or have explicit permission to assess, and
   it may be wrong.
4. **Outputs can be incorrect.** Verify facts, commands and explanations independently.
5. **Provenance matters.** Models are only as trustworthy as their data. Every release pins
   its dataset version and sources ([DATA_SOURCES.md](../DATA_SOURCES.md)).
6. **Capabilities and risks change between releases.** Read the limitations and experiment
   records of the specific version you use.
7. **Tools stay mediated.** When KittyLM runs inside KittyOS, it never acts on the operating
   system directly; tool access goes through the KittyOS tool layer and its validation.
