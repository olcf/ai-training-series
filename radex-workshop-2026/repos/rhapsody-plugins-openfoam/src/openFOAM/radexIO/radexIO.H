/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           |
     \\/     M anipulation  |
-------------------------------------------------------------------------------
Description
    Shared access to a radex key-value store.

    Owns the pieces that must be identical everywhere in a running process:
      - the radex client connection
      - the backend ("dragon" or "redis")
      - the identifier prefixed to every key
      - the key naming convention
            <identifier>_<name>_<timeStep>_<subdomainId>
        or, when the identifier is empty,
            <name>_<timeStep>_<subdomainId>

    All radexIO instances in a process share one connection: the first to be
    constructed opens it, the rest attach to it, and it is closed when the
    last instance is destroyed. Instances that ask for a different backend or
    identifier are warned and given the established one, so that function
    objects and solver code cannot disagree about where data goes or what it
    is called.

    This class is deliberately not a function object, so that it can be used
    directly from solver code for fields that never enter the objectRegistry.

\*---------------------------------------------------------------------------*/

#ifndef radexIO_H
#define radexIO_H

#include "Time.H"
#include "word.H"
#include "volFields.H"

#include "radex/handles.hpp"

#include <memory>
#include <string>
#include <type_traits>

namespace radex { class IClient; }

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                           Class radexIO Declaration
\*---------------------------------------------------------------------------*/

class radexIO
{
    // Private Data

        //- Shared with every other radexIO instance in this process
        std::shared_ptr<radex::IClient> client_;


    // Private Member Functions

        //- Connect, or attach to the connection already in use
        static std::shared_ptr<radex::IClient> acquireClient(const word& backend);

        //- Element type stored by put(). A tensor found at the other
        //- floating-point width is converted on read rather than rejected.
        typedef typename std::conditional
        <
            sizeof(scalar) == sizeof(double), float, double
        >::type otherScalar;

        //- Send nElements * nComponents contiguous scalars
        void putTensor
        (
            const radex::data::OutgoingHandle& handle,
            const scalar* data,
            label nElements,
            label nComponents
        ) const;

        //- Fetch a tensor, converting from the other floating-point width if
        //- that is what is stored. False if absent and not waiting.
        bool fetchTensor
        (
            const radex::data::IncomingHandle& handle,
            bool wait,
            scalar timeout,
            List<scalar>& data,
            List<label>& shape
        ) const;

        //- True if shape describes [nElements] or [nElements, nComponents].
        //- A trailing singleton dimension is accepted with a warning.
        bool shapeMatches
        (
            const List<label>& shape,
            label nElements,
            label nComponents,
            const std::string& key
        ) const;


public:

    // Constructors

        //- Construct with an explicit backend and identifier.
        //- An empty identifier leaves the established one untouched.
        radexIO(const word& backend, const word& identifier);

        //- Construct from system/radexDict, falling back to the defaults
        //- if that dictionary is absent
        explicit radexIO(const Time& runTime);

        //- No copy construct
        radexIO(const radexIO&) = delete;

        //- No copy assignment
        void operator=(const radexIO&) = delete;


    //- Destructor
    ~radexIO();


    // Member Functions

        //- The shared radex client
        radex::IClient& client() const;

        //- Backend in use by this process
        static const word& backend();

        //- Identifier prefixed to every key
        static const word& identifier();

        //- Establish the identifier if it is not already set. An empty word
        //- is ignored; a conflicting one warns and is discarded.
        static void setIdentifier(const word& identifier);

        //- Subdomain (MPI rank) of this process
        static label subdomainId();

        //- Build <identifier>_<name>_<timeName>_<subdomainId>
        //- or <name>_<timeName>_<subdomainId>
        std::string makeKey
        (
            const word& name,
            const std::string& timeName,
            label subdomainId
        ) const;


    // Transfer

        //- Send a Field as a tensor of shape [size] or [size, nComponents]
        template<class Type>
        void put
        (
            const radex::data::OutgoingHandle& handle,
            const Field<Type>& field
        ) const;

        //- Send the internal field of a volField
        template<class Type>
        void put
        (
            const radex::data::OutgoingHandle& handle,
            const GeometricField<Type, fvPatchField, volMesh>& field
        ) const;

        //- Send a single scalar
        void put
        (
            const radex::data::OutgoingHandle& handle,
            scalar value
        ) const;

        //- Copy a stored tensor into field, which must already be sized.
        //- False, with a warning, if the shape does not match or if the key
        //- is absent and wait is false.
        template<class Type>
        bool get
        (
            const radex::data::IncomingHandle& handle,
            Field<Type>& field,
            bool wait,
            scalar timeout
        ) const;

        //- As above, then correct the boundary conditions
        template<class Type>
        bool get
        (
            const radex::data::IncomingHandle& handle,
            GeometricField<Type, fvPatchField, volMesh>& field,
            bool wait,
            scalar timeout
        ) const;

        //- Copy a stored scalar into value
        bool get
        (
            const radex::data::IncomingHandle& handle,
            scalar& value,
            bool wait,
            scalar timeout
        ) const;
};


/*---------------------------------------------------------------------------*\
                          Class radexKey Declaration
\*---------------------------------------------------------------------------*/

//- Assembles store keys, always prefixed with the process-wide identifier.
//
//  The default convention is
//      radexKey(name).withTime(timeName).withSubdomain(rank)
//  Components may be left out, for instance to write a slot that is
//  overwritten rather than accumulated, and an index may be appended for data
//  written more than once per time step.
class radexKey
{
    // Private Data

        std::string key_;


public:

    // Constructors

        //- <identifier>_<name>, or <name> when no identifier is set
        explicit radexKey(const word& name);


    // Member Functions

        //- Append the time name
        radexKey& withTime(const std::string& timeName);

        //- Append the subdomain (MPI rank)
        radexKey& withSubdomain(label subdomainId);

        //- Append an index, e.g. an outer-corrector number
        radexKey& withIndex(label index);

        //- The assembled key
        const std::string& str() const { return key_; }

        //- The assembled key as a write handle
        radex::data::OutgoingHandle outgoing() const;

        //- The assembled key as a read handle
        radex::data::IncomingHandle incoming() const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

#ifdef NoRepository
    #include "radexIOTemplates.C"
#endif

#endif

// ************************************************************************* //
