using Test
using BPDecoderPlus
using TensorQEC

@testset "BPDecoderPlus.jl" begin

    @testset "Surface Code Creation" begin
        # Test surface code creation via TensorQEC
        code = SurfaceCode(3, 3)
        @test code isa SurfaceCode

        code = SurfaceCode(5, 5)
        @test code isa SurfaceCode

        # Create tanner graph
        tanner = CSSTannerGraph(code)
        @test tanner isa CSSTannerGraph
    end

    @testset "Error Models" begin
        n_qubits = 9

        # Test iid_error creation (single probability)
        em = iid_error(0.1, n_qubits)
        @test em !== nothing

        # Test depolarizing error model (px, py, pz)
        em = iid_error(0.05, 0.05, 0.05, n_qubits)
        @test em !== nothing

        # Test error pattern sampling
        ep = random_error_pattern(em)
        @test ep !== nothing
    end

    @testset "IP Decoder" begin
        code = SurfaceCode(3, 3)
        tanner = CSSTannerGraph(code)
        em = iid_error(0.05, 0.05, 0.05, 9)

        # Create decoder
        decoder = IPDecoder()
        @test decoder isa IPDecoder

        # Compile decoder
        compiled = compile(decoder, tanner)
        @test compiled !== nothing

        # Test decoding
        ep = random_error_pattern(em)
        syn = syndrome_extraction(ep, tanner)
        result = decode(compiled, syn)

        # Verify syndrome is satisfied
        @test syn == syndrome_extraction(result.error_pattern, tanner)
    end

    @testset "BP Decoder" begin
        code = SurfaceCode(3, 3)
        tanner = CSSTannerGraph(code)
        em = iid_error(0.05, 0.05, 0.05, 9)

        # BP without OSD
        decoder = BPDecoder(100, false)
        compiled = compile(decoder, tanner)

        ep = random_error_pattern(em)
        syn = syndrome_extraction(ep, tanner)
        result = decode(compiled, syn)

        # BP may not always succeed, but should return a result
        @test result !== nothing

        # BP with OSD (should always satisfy syndrome)
        decoder_osd = BPDecoder(100, true)
        compiled_osd = compile(decoder_osd, tanner)

        result_osd = decode(compiled_osd, syn)
        @test syn == syndrome_extraction(result_osd.error_pattern, tanner)
    end

    @testset "Logical Error Check" begin
        code = SurfaceCode(3, 3)
        tanner = CSSTannerGraph(code)
        em = iid_error(0.05, 0.05, 0.05, 9)

        ep = random_error_pattern(em)

        decoder = IPDecoder()
        compiled = compile(decoder, tanner)

        syn = syndrome_extraction(ep, tanner)
        result = decode(compiled, syn)

        # has_logical_error should return a Bool
        err = has_logical_error(tanner, ep, result.error_pattern)
        @test err isa Bool

        # Also test the underlying TensorQEC function
        lx, lz = logical_operator(tanner)
        err2 = check_logical_error(ep, result.error_pattern, lx, lz)
        @test err2 isa Bool
        @test err == err2
    end

    @testset "Quick Benchmark" begin
        result = quick_benchmark(distance=3, p=0.05, n_trials=10, decoder=:IP)

        @test haskey(result, "logical_error_rate")
        @test haskey(result, "std_error")
        @test result["distance"] == 3
        @test result["p"] == 0.05
        @test result["n_trials"] == 10
        @test 0 <= result["logical_error_rate"] <= 1
    end

    @testset "Atom Loss Model" begin
        # Test atom loss model creation
        loss_model = AtomLossModel(0.02)
        @test loss_model.p_loss == 0.02
        @test loss_model.detected == true

        loss_model = AtomLossModel(0.01, detected=false)
        @test loss_model.detected == false

        # Test apply_atom_loss
        code = SurfaceCode(3, 3)
        tanner = CSSTannerGraph(code)
        _, lost_qubits = apply_atom_loss(tanner, AtomLossModel(0.5))
        @test lost_qubits isa Vector{Int}
    end

    @testset "Create Decoder Helper" begin
        @test BPDecoderPlus.create_decoder(:IP) isa IPDecoder
        @test BPDecoderPlus.create_decoder(:BP) isa BPDecoder
        @test BPDecoderPlus.create_decoder(:BPOSD) isa BPDecoder

        @test_throws Exception BPDecoderPlus.create_decoder(:Unknown)
    end

    @testset "I/O Functions" begin
        using JSON

        # Test save and load
        test_data = Dict("test" => 123, "nested" => Dict("a" => 1))
        test_file = tempname() * ".json"

        save_results(test_data, test_file)
        @test isfile(test_file)

        loaded = load_results(test_file)
        @test loaded["test"] == 123
        @test loaded["nested"]["a"] == 1

        rm(test_file)
    end

end

println("\nAll tests passed!")
