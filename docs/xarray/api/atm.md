# Atmosphere

!!! info "Planned namespace"
    `xrtoolz.atm` is a reserved namespace for atmospheric physics. The
    planned surface includes potential temperature and wind speed /
    direction, with trace-gas (methane) physics under `xrtoolz.atm.gas.ch4`
    (column averaging kernel, dry-air column, mixing ratio). It exports
    nothing yet — this page will fill in as operators land.

The design rule still holds: only true physics lives here; anything
domain-agnostic belongs in [`geo`](geo/coords.md).

::: xrtoolz.atm
